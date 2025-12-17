import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import py_vollib_vectorized
import os
from tqdm import tqdm
import warnings

# --- CONFIGURARE ---
RAW_FILE = 'options_eod_all.csv'
HISTORY_FILE = 'SPX_Advanced_History_2012_2025.csv'
OUTPUT_FOLDER = 'vol_frames'
CHUNK_SIZE = 100000

# Limite fixe pentru axe (ca animația să fie stabilă vizual)
Z_LIMITS = (0.05, 0.60)  # IV între 5% și 60%
M_LIMITS = (0.8, 1.15)  # Moneyness
T_LIMITS = (0.05, 1.5)  # Timp

warnings.filterwarnings('ignore')


def setup_environment():
    """Creeaza folderul de output si incarca istoricul calculat."""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"--- Incarc parametrii istorici (r, q) din {HISTORY_FILE} ---")
    try:
        hist_df = pd.read_csv(HISTORY_FILE)
        # FIX 1: Normalizare date pentru lookup precis
        hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.normalize()

        # Dictionar lookup rapid
        param_map = {}
        for _, row in hist_df.iterrows():
            param_map[row['Date']] = (
                row['Risk_Free_Rate'],
                row['Dividend_Yield_Continuous'],
                row['Underlying_Price']
            )
        return param_map
    except Exception as e:
        print(f"Eroare critica la incarcarea istoricului: {e}")
        exit()


def process_and_plot_day(day_df, date_ts, params, frame_idx):
    """
    Genereaza imaginea pentru o singura zi.
    Returneaza True daca a salvat imaginea, False altfel.
    """
    # Lucram pe o copie
    day_df = day_df.copy()

    r, q, S_hist = params

    # Folosim S din raw data daca e valid, altfel fallback la istoric
    S_raw = day_df['underlying_bid_1545'].iloc[0]
    S = S_raw if S_raw > 0 else S_hist

    # Pre-calcule
    day_df['expiration'] = pd.to_datetime(day_df['expiration'])
    day_df['T'] = (day_df['expiration'] - day_df['quote_date']).dt.days / 365.0

    # Calcul Mid Price inainte de filtre
    day_df['mid_price'] = (day_df['bid_1545'] + day_df['ask_1545']) / 2

    # FIX 5: Filtru ask >= bid (acceptam spread 0)
    valid = (day_df['ask_1545'] >= day_df['bid_1545']) & (day_df['mid_price'] > 0.05)
    df = day_df[valid].copy()

    # Filtru zona de interes
    df = df[(df['T'] >= T_LIMITS[0]) & (df['T'] <= T_LIMITS[1])]

    if df.empty: return False

    # Vectorized IV
    df['flag'] = df['option_type'].str.lower().str.get(0)

    # FIX 2: S ca array pentru stabilitate broadcasting
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

    # Selectie OTM (Smile logic)
    puts = df[(df['flag'] == 'p') & (df['strike'] <= S)]
    calls = df[(df['flag'] == 'c') & (df['strike'] > S)]
    surf_df = pd.concat([puts, calls])

    if surf_df.empty: return False

    surf_df['Moneyness'] = surf_df['strike'] / S
    surf_df = surf_df[(surf_df['Moneyness'] >= M_LIMITS[0]) & (surf_df['Moneyness'] <= M_LIMITS[1])]

    if len(surf_df) < 20: return False

    # Interpolare
    x = surf_df['Moneyness']
    y = surf_df['T']
    z = surf_df['IV']

    xi = np.linspace(M_LIMITS[0], M_LIMITS[1], 30)
    yi = np.linspace(T_LIMITS[0], T_LIMITS[1], 30)
    X, Y = np.meshgrid(xi, yi)

    try:
        # FIX 3: Combinatie Linear + Nearest pentru a umple golurile (NaN)
        Z = griddata((x, y), z, (X, Y), method='linear')
        Z_nn = griddata((x, y), z, (X, Y), method='nearest')
        # Unde linear da NaN (pe margini), punem nearest
        Z = np.where(np.isnan(Z), Z_nn, Z)
    except:
        return False

    # Plotare
    fig = plt.figure(figsize=(10, 7), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.9)

    ax.set_zlim(Z_LIMITS)
    ax.set_ylim(T_LIMITS)
    ax.set_xlim(M_LIMITS)
    ax.invert_xaxis()

    date_str = date_ts.strftime('%Y-%m-%d')
    ax.set_title(f'SPX Volatility Surface | {date_str}\nSpot: {S:.0f} | r: {r:.1%} | q: {q:.1%}', fontsize=12)
    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Years to Expiry')
    ax.set_zlabel('IV')
    ax.view_init(elev=25, azim=-120)

    filename = f"{OUTPUT_FOLDER}/frame_{frame_idx:05d}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)  # Cleanup memorie

    return True  # FIX 4: Confirmam succesul


# --- MAIN LOOP ---
if __name__ == "__main__":
    param_map = setup_environment()

    print("Start generare frame-uri...")
    reader = pd.read_csv(RAW_FILE, chunksize=CHUNK_SIZE)
    buffer_df = pd.DataFrame()

    frame_count = 0

    try:
        for chunk in tqdm(reader, desc="Generating Frames"):
            chunk['quote_date'] = pd.to_datetime(chunk['quote_date'])

            # Buffer logic
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

            # Procesare zile unice
            unique_days = chunk['quote_date'].unique()
            for day in unique_days:
                # FIX 1: Normalizare in loop
                ts = pd.Timestamp(day).normalize()

                if ts in param_map:
                    params = param_map[ts]
                    day_data = chunk[chunk['quote_date'] == day]

                    # FIX 4: Incrementam DOAR daca s-a salvat imaginea
                    if process_and_plot_day(day_data, ts, params, frame_count):
                        frame_count += 1

    except KeyboardInterrupt:
        print("\nOprit de utilizator.")

    print(f"\nDone! {frame_count} imagini salvate valid in '{OUTPUT_FOLDER}/'.")