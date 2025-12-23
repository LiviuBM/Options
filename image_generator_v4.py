import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import py_vollib_vectorized
import os
import shutil
from tqdm import tqdm
import warnings

# --- CONFIGURARE ---
RAW_FILE = 'options_eod_all.csv'
HISTORY_FILE = 'SPX_Advanced_History_Refined_v2.csv'  # Fisierul cu Rata Kalman
OUTPUT_FOLDER = 'frames_output_v2'

# LIMITE FIXE PENTRU ANIMATIE (CRITIC!)
# Acestea asigura ca axele nu se misca de la o zi la alta
Z_LIMITS = (0.0, 0.60)  # IV intre 0% si 60%
M_LIMITS = (0.75, 1.25)  # Moneyness (K/F) intre 0.75 si 1.25
T_LIMITS = (0.05, 1.5)  # Timp: 2 saptamani pana la 1.5 ani

CHUNK_SIZE = 100000

warnings.filterwarnings('ignore')


# --- FILTRE "VALIDATE" (literatura) ---
#  - Bid/ask strict valide: bid>0, ask>bid (Carr & Wu, 2009; Dumas et al., 1998)
#  - Bound intrinsec: mid >= intrinsic (no-arbitrage basic)
#  - Vega cutoff: vega >= 0.5 (OptionMetrics IvyDB manual)
MIN_T_DAYS = 7
MIN_T = MIN_T_DAYS / 365.0
VEGA_CUTOFF = 0.5

def _bs_vega(S, K, T, r, q, sigma):
    sigma = np.maximum(sigma, 1e-8)
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    return S * np.exp(-q * T) * np.sqrt(T) * phi

def _intrinsic_value(S, K, flag):
    call = np.maximum(0.0, S - K)
    put = np.maximum(0.0, K - S)
    return np.where(flag == 'c', call, put)

def setup_environment():
    # Curatare/Creare folder output
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Incarcare Parametri Istorici
    print(f"--- Incarc parametrii istorici din {HISTORY_FILE} ---")
    try:
        hist_df = pd.read_csv(HISTORY_FILE)
        hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.normalize()

        # Cream un map: Date -> (R_Kalman, Q_Continuous, Spot)
        param_map = {}
        for _, row in hist_df.iterrows():
            param_map[row['Date']] = (
                row['Risk_Free_Rate_Kalman'],
                row['Dividend_Yield_Continuous'],
                row['Underlying_Price']
            )
        return param_map
    except Exception as e:
        print(f"Eroare critica la citirea istoricului: {e}")
        exit()


def process_and_save_frame(day_df, date_ts, params, frame_idx):
    """Proceseaza o singura zi si salveaza imaginea (Versiunea Anti-Spike)."""

    r, q, S_hist = params

    # 1. Preparare Date
    day_df = day_df.copy()
    day_df['expiration'] = pd.to_datetime(day_df['expiration'])
    day_df['T'] = (day_df['expiration'] - day_df['quote_date']).dt.days / 365.0
    day_df['mid_price'] = (day_df['bid_1545'] + day_df['ask_1545']) / 2

    # S poate lipsi din raw data, il luam din istoric
    S_raw = day_df['underlying_bid_1545'].iloc[0]
    S = S_raw if S_raw > 0 else S_hist

    # 2. Filtre de Calitate Initiale
    day_df['spread_pct'] = (day_df['ask_1545'] - day_df['bid_1545']) / day_df['mid_price']

    valid = (
            (day_df['ask_1545'] >= day_df['bid_1545']) &
            (day_df['mid_price'] > 0.05) &
            (day_df['T'] >= T_LIMITS[0]) &
            (day_df['T'] <= T_LIMITS[1])
    )
    df = day_df[valid].copy()

    if df.empty or len(df) < 20: return False

    # 3. Calcul IV Vectorizat
    df['flag'] = df['option_type'].str.lower().str.get(0)
    S_arr = np.full(len(df), S)

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

    # Selectie OTM Only
    puts = df[(df['flag'] == 'p') & (df['strike'] <= S)]
    calls = df[(df['flag'] == 'c') & (df['strike'] > S)]
    surf_df = pd.concat([puts, calls])

    if surf_df.empty: return False

    # 4. Forward Moneyness (K / F)
    # Calculam F explicit in coloana separata (repara KeyError 'F')
    surf_df['F'] = S * np.exp((r - q) * surf_df['T'])
    surf_df['Moneyness'] = surf_df['strike'] / surf_df['F']

    # --- FILTRE AVANSATE ANTI-SPIKE ---

    # A. Filtru Spread Contextual
    # ATM (0.95-1.05) trebuie sa fie curat (<15% spread)
    # Wings pot fi mai zgomotoase (<40%)
    is_atm = (surf_df['Moneyness'] >= 0.95) & (surf_df['Moneyness'] <= 1.05)
    surf_df = surf_df[
        (~is_atm & (surf_df['spread_pct'] < 0.40)) |
        (is_atm & (surf_df['spread_pct'] < 0.15))
        ]

    # B. Filtrare Limite Axe
    surf_df = surf_df[
        (surf_df['Moneyness'] >= M_LIMITS[0]) &
        (surf_df['Moneyness'] <= M_LIMITS[1]) &
        (surf_df['IV'] < Z_LIMITS[1])
        ]

    if len(surf_df) < 20: return False

    # C. Quantile Clipping (Taie spike-urile in jos)
    # Eliminam cele mai mici 2% valori IV (de obicei erori)
    if len(surf_df) > 50:
        q_low = surf_df['IV'].quantile(0.02)
        surf_df = surf_df[surf_df['IV'] >= q_low]
    # ----------------------------------

    # 5. Interpolare si Smoothing
    x = surf_df['Moneyness']
    y = surf_df['T']
    z = surf_df['IV']

    xi = np.linspace(M_LIMITS[0], M_LIMITS[1], 40)
    yi = np.linspace(T_LIMITS[0], T_LIMITS[1], 40)
    X, Y = np.meshgrid(xi, yi)

    try:
        Z = griddata((x, y), z, (X, Y), method='linear')

        # Nearest neighbor pentru gauri mici
        Z_nn = griddata((x, y), z, (X, Y), method='nearest')
        Z = np.where(np.isnan(Z), Z_nn, Z)

        # Gaussian Smoothing (0.8 pentru eliminare artefacte ramase)
        Z = gaussian_filter(Z, sigma=0.8)
    except:
        return False

    # 6. Generare Plot
    fig = plt.figure(figsize=(10, 7), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.9, vmin=Z_LIMITS[0],
                           vmax=Z_LIMITS[1])

    # Axe Fixe
    ax.set_zlim(Z_LIMITS)
    ax.set_ylim(T_LIMITS)
    ax.set_xlim(M_LIMITS)
    ax.invert_xaxis()

    date_str = date_ts.strftime('%Y-%m-%d')
    ax.set_title(
        f'SPX Vol Surface | {date_str}\n'
        f'Spot: {S:.0f} | r(Kalman): {r:.2%} | q(Cont): {q:.2%}',
        fontsize=12
    )
    ax.set_xlabel('Fwd Moneyness (K/F)')
    ax.set_ylabel('Years to Expiry')
    ax.set_zlabel('Implied Volatility')

    # Unghi vizualizare
    ax.view_init(elev=25, azim=-120)

    filename = f"{OUTPUT_FOLDER}/frame_{frame_idx:05d}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)  # Eliberare memorie

    return True


# --- MAIN LOOP ---
if __name__ == "__main__":
    param_map = setup_environment()

    print("Start generare cadre...")
    reader = pd.read_csv(RAW_FILE, chunksize=CHUNK_SIZE)
    buffer_df = pd.DataFrame()
    frame_count = 0

    try:
        for chunk in tqdm(reader, desc="Processing chunks"):
            chunk['quote_date'] = pd.to_datetime(chunk['quote_date'])

            # --- Buffer Logic ---
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

            # Procesare pe zile
            unique_days = chunk['quote_date'].unique()

            # --- FIX: Sortare Numpy ---
            unique_days = np.sort(unique_days)

            for day in unique_days:
                ts = pd.Timestamp(day).normalize()

                if ts in param_map:
                    day_data = chunk[chunk['quote_date'] == day]
                    if process_and_save_frame(day_data, ts, param_map[ts], frame_count):
                        frame_count += 1

    except KeyboardInterrupt:
        print("\nOprit de utilizator. Fisierele salvate sunt valide.")

    print(f"\nGenerare completa! {frame_count} frame-uri salvate in '{OUTPUT_FOLDER}/'.")