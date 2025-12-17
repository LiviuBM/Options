import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter  # <--- MAGIC TOUCH
import py_vollib_vectorized
import os
from tqdm import tqdm
import warnings

# --- CONFIGURARE ---
RAW_FILE = 'options_eod_all.csv'
HISTORY_FILE = 'SPX_Advanced_History_2012_2025.csv'
OUTPUT_FOLDER = 'vol_frames_final'  # Folder nou
CHUNK_SIZE = 100000

# Limite fixe (ajustate pentru Forward Moneyness)
Z_LIMITS = (0.05, 0.60)  # IV
M_LIMITS = (0.8, 1.15)  # Moneyness (K/F) - Acum e centrat perfect pe 1
T_LIMITS = (0.05, 1.5)  # Timp

warnings.filterwarnings('ignore')


def setup_environment():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"--- Incarc parametrii istorici (r, q) din {HISTORY_FILE} ---")
    try:
        hist_df = pd.read_csv(HISTORY_FILE)
        hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.normalize()
        param_map = {}
        for _, row in hist_df.iterrows():
            param_map[row['Date']] = (
                row['Risk_Free_Rate'],
                row['Dividend_Yield_Continuous'],
                row['Underlying_Price']
            )
        return param_map
    except Exception as e:
        print(f"Eroare critica: {e}")
        exit()


def process_and_plot_day(day_df, date_ts, params, frame_idx):
    day_df = day_df.copy()
    r, q, S_hist = params

    S_raw = day_df['underlying_bid_1545'].iloc[0]
    S = S_raw if S_raw > 0 else S_hist

    # 1. Calcul T si Mid Price
    day_df['expiration'] = pd.to_datetime(day_df['expiration'])
    day_df['T'] = (day_df['expiration'] - day_df['quote_date']).dt.days / 365.0
    day_df['mid_price'] = (day_df['bid_1545'] + day_df['ask_1545']) / 2

    # 2. CALITATE: Filtru Spread Relativ
    # Eliminam optiunile unde spread-ul e > 40% din pret (zgomot pur)
    # Folosim 0.4 ca sa nu pierdem wings in crize, dar sa scapam de aberatii
    day_df['spread_pct'] = (day_df['ask_1545'] - day_df['bid_1545']) / day_df['mid_price']

    valid = (
            (day_df['ask_1545'] >= day_df['bid_1545']) &
            (day_df['mid_price'] > 0.05) &
            (day_df['spread_pct'] < 0.4)  # <--- FILTRU NOU
    )
    df = day_df[valid].copy()

    df = df[(df['T'] >= T_LIMITS[0]) & (df['T'] <= T_LIMITS[1])]
    if df.empty: return False

    # 3. Calcul IV
    df['flag'] = df['option_type'].str.lower().str.get(0)
    S_arr = np.full(len(df), S, dtype=float)

    try:
        df['IV'] = py_vollib_vectorized.vectorized_implied_volatility(
            price=df['mid_price'].values,
            S=S_arr,
            K=df['strike'].values,
            t=df['T'].values,
            r=r,
            q=q,
            flag=df['flag'].values,
            return_as='numpy',
            on_error='ignore'
        )
    except:
        return False

    df = df.dropna(subset=['IV'])

    # Selectie OTM (Puts < S, Calls > S)
    puts = df[(df['flag'] == 'p') & (df['strike'] <= S)]
    calls = df[(df['flag'] == 'c') & (df['strike'] > S)]
    surf_df = pd.concat([puts, calls])

    if surf_df.empty: return False

    # 4. UPGRADE: Forward Moneyness (K / F)
    # F = S * e^((r-q)T)
    # Asta aliniaza "valea" la 1.0 indiferent de dobanzi
    surf_df['F'] = S * np.exp((r - q) * surf_df['T'])
    surf_df['Moneyness'] = surf_df['strike'] / surf_df['F']

    surf_df = surf_df[(surf_df['Moneyness'] >= M_LIMITS[0]) & (surf_df['Moneyness'] <= M_LIMITS[1])]
    if len(surf_df) < 20: return False

    # 5. Interpolare
    x = surf_df['Moneyness']
    y = surf_df['T']
    z = surf_df['IV']

    xi = np.linspace(M_LIMITS[0], M_LIMITS[1], 30)  # Rezolutie
    yi = np.linspace(T_LIMITS[0], T_LIMITS[1], 30)
    X, Y = np.meshgrid(xi, yi)

    try:
        Z = griddata((x, y), z, (X, Y), method='linear')
        Z_nn = griddata((x, y), z, (X, Y), method='nearest')
        Z = np.where(np.isnan(Z), Z_nn, Z)

        # 6. UPGRADE: Gaussian Smoothing
        # Sigma mic (0.5) netezeste doar zgomotul, pastreaza forma
        Z = gaussian_filter(Z, sigma=0.5)

    except:
        return False

    # 7. Plotare
    fig = plt.figure(figsize=(10, 7), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.9)  # Antialiased True

    ax.set_zlim(Z_LIMITS)
    ax.set_ylim(T_LIMITS)
    ax.set_xlim(M_LIMITS)
    ax.invert_xaxis()

    date_str = date_ts.strftime('%Y-%m-%d')
    # Adaugam info despre regim (Fwd Price) in titlu
    ax.set_title(f'SPX Vol Surface (Forward Moneyness) | {date_str}\nSpot: {S:.0f} | r: {r:.1%} | q: {q:.1%}',
                 fontsize=12)
    ax.set_xlabel('Fwd Moneyness (K/F)')
    ax.set_ylabel('Years to Expiry')
    ax.set_zlabel('IV')
    ax.view_init(elev=25, azim=-120)

    filename = f"{OUTPUT_FOLDER}/frame_{frame_idx:05d}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

    return True


# --- MAIN LOOP ---
if __name__ == "__main__":
    param_map = setup_environment()
    print("Start generare (Spread Filter + Fwd Moneyness + Smoothing)...")

    reader = pd.read_csv(RAW_FILE, chunksize=CHUNK_SIZE)
    buffer_df = pd.DataFrame()
    frame_count = 0

    try:
        for chunk in tqdm(reader, desc="Processing"):
            chunk['quote_date'] = pd.to_datetime(chunk['quote_date'])

            if not buffer_df.empty:
                chunk = pd.concat([buffer_df, chunk])
                buffer_df = pd.DataFrame()

            last_day = chunk['quote_date'].max()
            if chunk['quote_date'].nunique() > 1:
                mask_last = chunk['quote_date'] == last_day
                buffer_df = chunk[mask_last].copy()
                chunk = chunk[~mask_last].copy()
            else:
                buffer_df = pd.concat([buffer_df, chunk])
                continue

            unique_days = chunk['quote_date'].unique()
            for day in unique_days:
                ts = pd.Timestamp(day).normalize()
                if ts in param_map:
                    if process_and_plot_day(chunk[chunk['quote_date'] == day], ts, param_map[ts], frame_count):
                        frame_count += 1

    except KeyboardInterrupt:
        print("\nOprit.")

    print(f"\nDone! {frame_count} frames saved in '{OUTPUT_FOLDER}/'.")