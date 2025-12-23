import pandas as pd
import numpy as np
import py_vollib_vectorized
from scipy.stats import linregress
from tqdm import tqdm
import warnings
import os

# --- CONFIGURARE ---
RAW_FILE = 'options_eod_all.csv'
YIELD_FILE = 'S&P500Index-Returns.xlsx'
OUTPUT_FILE = 'SPX_Advanced_History_Refined_v2.csv'
CHUNK_SIZE = 100000

# Coloane din fisierul de randamente
COL_DATE = 'Calendar Date'
COL_RET_WITH_DIV = 'Value-Weighted Return-incl. dividends'
COL_RET_NO_DIV = 'Value-Weighted Return-excl. dividends'

warnings.filterwarnings('ignore')


# --- FILTRE "VALIDATE" (literatura) ---
#  - Bid/ask strict valide: bid>0, ask>bid (Carr & Wu, 2009; Dumas et al., 1998)
#  - Bound intrinsec: mid >= intrinsic (no-arbitrage basic)
#  - Trunchiere wings pe regula "doua zero-bid consecutiv" (Cboe VIX methodology)
#  - Vega cutoff: vega >= 0.5 (OptionMetrics IvyDB manual)
MIN_T_DAYS = 7  # Carr & Wu (2009) folosesc >= 7 zile calendaristice
MIN_T = MIN_T_DAYS / 365.0
VEGA_CUTOFF = 0.5

def _bs_vega(S, K, T, r, q, sigma):
    """Dollar vega (dPrice/dSigma), Black-Scholes-Merton."""
    # Evitam impartiri la zero
    sigma = np.maximum(sigma, 1e-8)
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    return S * np.exp(-q * T) * np.sqrt(T) * phi

def _intrinsic_value(S, K, flag):
    # flag: 'c' sau 'p'
    call = np.maximum(0.0, S - K)
    put = np.maximum(0.0, K - S)
    return np.where(flag == 'c', call, put)

def _vix_style_wing_selection(df_one_exp, F, K0):
    """Implementeaza regula Cboe: exclude zero-bid/zero-ask si opreste dupa 2 consecutive."""
    # Intoarce un mask boolean cu optiunile selectate (OTM puts + OTM calls + strike=K0 put&call daca exista)
    sel = np.zeros(len(df_one_exp), dtype=bool)

    # indexare pentru mapping
    df = df_one_exp.sort_values('strike').reset_index()
    idx = df['index'].values

    strikes = df['strike'].values
    bids = df['bid_1545'].values
    asks = df['ask_1545'].values
    flags = df['flag'].values

    # Put OTM: strike < K0
    put_mask = (flags == 'p') & (strikes < K0)
    put_df = df[put_mask].sort_values('strike', ascending=False)

    zero_run = 0
    for _, row in put_df.iterrows():
        if row['bid_1545'] == 0 or row['ask_1545'] == 0:
            zero_run += 1
            if zero_run >= 2:
                break
            continue
        zero_run = 0
        sel[df['index'] == row['index']] = True

    # Call OTM: strike > K0
    call_mask = (flags == 'c') & (strikes > K0)
    call_df = df[call_mask].sort_values('strike', ascending=True)

    zero_run = 0
    for _, row in call_df.iterrows():
        if row['bid_1545'] == 0 or row['ask_1545'] == 0:
            zero_run += 1
            if zero_run >= 2:
                break
            continue
        zero_run = 0
        sel[df['index'] == row['index']] = True

    # Strike = K0: include at-the-money put+call daca exista si au quote valide
    atm_df = df[df['strike'] == K0]
    if not atm_df.empty:
        for _, row in atm_df.iterrows():
            if row['bid_1545'] > 0 and row['ask_1545'] > row['bid_1545']:
                sel[df['index'] == row['index']] = True

    # Convertim inapoi la index-ul original
    out = pd.Series(False, index=df_one_exp.index)
    out.loc[idx[sel]] = True
    return out.values

# --- 1. CLASA KALMAN FILTER (Doar pentru Rate) ---
class KalmanFilter1D:
    def __init__(self, initial_state, process_noise, measurement_noise):
        """
        initial_state: Estimarea initiala a ratei (ex: 0.02)
        process_noise (Q): Cat de mult se schimba rata reala? (Mic, ex: 1e-6, rata are inertie)
        measurement_noise (R): Cat zgomot e in datele Put-Call? (Mare, ex: 5e-4, bid-ask spread)
        """
        self.state = initial_state
        self.P = 1.0
        self.Q = process_noise
        self.R = measurement_noise

    def update(self, measurement):
        # 1. PREDICTIE (Time Update)
        # Random Walk: Presupunem ca rata ramane constanta, dar incertitudinea creste
        self.P = self.P + self.Q

        if measurement is None or np.isnan(measurement):
            return self.state

        # 2. CORECTIE (Measurement Update)
        # Kalman Gain
        K = self.P / (self.P + self.R)

        # Update State
        self.state = self.state + K * (measurement - self.state)

        # Update Covariance
        self.P = (1 - K) * self.P

        return self.state


# --- 2. LOGICA DIVIDENDE (TR vs PR - Structurally Smoothed) ---
def load_dynamic_yields(yield_file_path):
    print(f"--- Incarc datele de dividende din: {yield_file_path} ---")

    if not os.path.exists(yield_file_path):
        print(f"EROARE: Fisierul {yield_file_path} nu exista.")
        return None

    try:
        if yield_file_path.endswith('.xlsx'):
            df = pd.read_excel(yield_file_path, engine='openpyxl')
        else:
            try:
                df = pd.read_csv(yield_file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(yield_file_path, encoding='cp1252')
    except Exception as e:
        print(f"EROARE CITIRE: {e}")
        return None

    df.columns = df.columns.str.strip()

    # Verificare coloane
    required = [COL_DATE, COL_RET_WITH_DIV, COL_RET_NO_DIV]
    if not all(col in df.columns for col in required):
        print(f"EROARE COLOANE. Am nevoie de: {required}")
        return None

    # Procesare
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df.sort_values(COL_DATE).set_index(COL_DATE)

    df[COL_RET_WITH_DIV] = pd.to_numeric(df[COL_RET_WITH_DIV], errors='coerce').fillna(0)
    df[COL_RET_NO_DIV] = pd.to_numeric(df[COL_RET_NO_DIV], errors='coerce').fillna(0)

    # Indici Sintetici
    df['TR_Index'] = (1 + df[COL_RET_WITH_DIV]).cumprod()
    df['PR_Index'] = (1 + df[COL_RET_NO_DIV]).cumprod()

    # Yield Calculation (Trailing 252 zile) - Smooth prin definitie
    window = 252
    df['Ratio'] = df['TR_Index'] / df['PR_Index']

    # Yield Discret
    df['yield_discrete'] = (df['Ratio'] / df['Ratio'].shift(window)) - 1

    # Umplem golurile (backfill pentru primul an)
    df['yield_discrete'] = df['yield_discrete'].fillna(method='bfill').interpolate()
    df['yield_discrete'] = df['yield_discrete'].clip(lower=0.0, upper=0.06)

    # Continuous Yield pentru Black-Scholes
    df['yield_continuous'] = np.log(1 + df['yield_discrete'])

    # Mapare cu data normalizata
    yield_map = {k.normalize(): v for k, v in df['yield_continuous'].to_dict().items()}

    print(f"Succes! Yield continuu mediu: {df['yield_continuous'].mean():.2%}")
    return yield_map


# --- 3. EXTRAGERE RISK FREE RATE (Raw Measurement) ---
def extract_implied_rf(day_df):
    if day_df.empty: return None
    S = day_df['underlying_bid_1545'].iloc[0]

    # ATM si T > 2 saptamani
    subset = day_df[
        (day_df['strike'] > S * 0.90) &
        (day_df['strike'] < S * 1.10) &
        (day_df['T'] > 0.04)
        ].copy()

    if subset.empty: return None

    calls = subset[subset['flag'] == 'c'][['expiration', 'strike', 'mid_price', 'T']]
    puts = subset[subset['flag'] == 'p'][['expiration', 'strike', 'mid_price']]
    merged = pd.merge(calls, puts, on=['expiration', 'strike'], suffixes=('_c', '_p'))

    if len(merged) < 5: return None
    merged['diff'] = merged['mid_price_c'] - merged['mid_price_p']

    rates = []
    for exp in merged['expiration'].unique():
        exp_data = merged[merged['expiration'] == exp]
        if len(exp_data) < 4: continue
        T = exp_data['T'].iloc[0]
        try:
            # Panta regresiei = -e^(-rT)
            slope, _, _, _, _ = linregress(exp_data['strike'], exp_data['diff'])
            if -1.15 <= slope <= -0.85:
                r_implied = -np.log(-slope) / T
                # Filtru grosier pentru aberatii extreme
                if -0.02 <= r_implied < 0.15:
                    rates.append(r_implied)
        except:
            continue

    return np.median(rates) if rates else None


# --- 4. CALCUL STATISTICI ZILNICE ---
def process_day_statistics(day_df, k_rate, raw_r, q_yield):
    try:
        # IV Vectorizat folosind parametrii optimizati
        day_df['IV'] = py_vollib_vectorized.vectorized_implied_volatility(
            price=day_df['mid_price'].values,
            S=day_df['underlying_bid_1545'].values,
            K=day_df['strike'].values,
            t=day_df['T'].values,
            r=k_rate,  # Folosim rata Kalman
            q=q_yield,  # Folosim yield-ul structural
            flag=day_df['flag'].values,
            return_as='numpy',
            on_error='ignore'
        )
    except Exception:
        return None

    # Curatare IV
    day_df = day_df.dropna(subset=['IV'])
    day_df = day_df[(day_df['IV'] > 0.01) & (day_df['IV'] < 3.0)]
    if day_df.empty: return None

    S = day_df['underlying_bid_1545'].iloc[0]

    # --- METODA NOUA: ATM Robust (Median in banda +/- 3%) ---
    atm_band = day_df[
        (day_df['strike'] >= S * 0.97) &
        (day_df['strike'] <= S * 1.03)
        ]

    if not atm_band.empty:
        atm_iv = atm_band['IV'].median()
    else:
        # Fallback (foarte rar)
        day_df['dist'] = (day_df['strike'] - S).abs()
        atm_iv = day_df.nsmallest(1, 'dist')['IV'].values[0]

    # Skew metrics
    put_90 = day_df[(day_df['strike'] >= S * 0.88) & (day_df['strike'] <= S * 0.92) & (day_df['flag'] == 'p')]
    call_110 = day_df[(day_df['strike'] >= S * 1.08) & (day_df['strike'] <= S * 1.12) & (day_df['flag'] == 'c')]

    put_80 = day_df[(day_df['strike'] >= S * 0.78) & (day_df['strike'] <= S * 0.82) & (day_df['flag'] == 'p')]
    call_120 = day_df[(day_df['strike'] >= S * 1.18) & (day_df['strike'] <= S * 1.22) & (day_df['flag'] == 'c')]

    iv_p90 = put_90['IV'].mean()
    iv_c110 = call_110['IV'].mean()
    iv_p80 = put_80['IV'].mean()
    iv_c120 = call_120['IV'].mean()

    skew = iv_p90 - iv_c110 if (not np.isnan(iv_p90) and not np.isnan(iv_c110)) else np.nan
    kurt_proxy = ((iv_p80 + iv_c120) / 2) - atm_iv if (not np.isnan(iv_p80) and not np.isnan(iv_c120)) else np.nan

    return {
        'Date': day_df['quote_date'].iloc[0],
        'Underlying_Price': S,
        'Risk_Free_Rate_Raw': raw_r if raw_r is not None else np.nan,
        'Risk_Free_Rate_Kalman': k_rate,
        'Dividend_Yield_Continuous': q_yield,
        'ATM_IV': atm_iv,
        'Skew_90_110': skew,
        'Kurtosis_Wings': kurt_proxy,
        'Options_Volume': len(day_df)
    }


# --- MAIN EXECUTION ---
if __name__ == "__main__":

    # 1. Incarcare Dividende (Structural Smoothed)
    yield_map = load_dynamic_yields(YIELD_FILE)

    # 2. Initializare Kalman Filter DOAR pentru Rata
    # Q (Process Noise) = 1e-6: Dobanda reala are inertie mare (nu sare zilnic)
    # R (Measurement Noise) = 5e-4: Put-Call Parity e zgomotos (bid-ask wide)
    kf_r = KalmanFilter1D(initial_state=0.015, process_noise=1e-6, measurement_noise=5e-4)

    print(f"Start procesare optiuni din: {RAW_FILE}")
    reader = pd.read_csv(RAW_FILE, chunksize=CHUNK_SIZE)
    results = []
    buffer_df = pd.DataFrame()

    try:
        for chunk in tqdm(reader, desc="Procesare"):
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

            # --- Pre-procesare Chunk ---
            chunk['expiration'] = pd.to_datetime(chunk['expiration'])
            chunk['T'] = (chunk['expiration'] - chunk['quote_date']).dt.days / 365.0

            # Filtru Integritate Preturi
            valid_prices = (chunk['ask_1545'] >= chunk['bid_1545'])
            chunk.loc[valid_prices, 'mid_price'] = (chunk.loc[valid_prices, 'bid_1545'] + chunk.loc[
                valid_prices, 'ask_1545']) / 2

            mask = valid_prices & (chunk['T'] > 0.005) & (chunk['mid_price'] > 0.05) & (
                        chunk['underlying_bid_1545'] > 0)
            valid_chunk = chunk[mask].copy()

            if valid_chunk.empty: continue
            valid_chunk['flag'] = valid_chunk['option_type'].str.lower().str.get(0)

            # --- Iterare Zile ---
            unique_days = valid_chunk['quote_date'].unique()
            for day in unique_days:
                day_data = valid_chunk[valid_chunk['quote_date'] == day].copy()
                ts_day = pd.to_datetime(day).normalize()

                # A. Risk Free Rate -> Raw + Kalman
                r_raw = extract_implied_rf(day_data)
                final_r = kf_r.update(r_raw)

                # B. Dividend Yield -> Direct din Index (deja smooth)
                # Fallback la 1.5% daca lipseste data din Excel
                q_val = yield_map.get(ts_day, 0.015) if yield_map else 0.015

                # C. Calcul Statistici
                stats = process_day_statistics(day_data, final_r, r_raw, q_val)

                if stats:
                    results.append(stats)

        # --- Procesare Final Buffer ---
        if not buffer_df.empty:
            buffer_df['expiration'] = pd.to_datetime(buffer_df['expiration'])
            buffer_df['T'] = (buffer_df['expiration'] - buffer_df['quote_date']).dt.days / 365.0
            valid_prices = (buffer_df['ask_1545'] >= buffer_df['bid_1545'])
            buffer_df.loc[valid_prices, 'mid_price'] = (buffer_df.loc[valid_prices, 'bid_1545'] + buffer_df.loc[
                valid_prices, 'ask_1545']) / 2

            mask = valid_prices & (buffer_df['T'] > 0.005) & (buffer_df['mid_price'] > 0.05)
            last_valid = buffer_df[mask].copy()

            if not last_valid.empty:
                last_valid['flag'] = last_valid['option_type'].str.lower().str.get(0)

                r_raw = extract_implied_rf(last_valid)
                final_r = kf_r.update(r_raw)

                ts_day = last_valid['quote_date'].iloc[0].normalize()
                q_val = yield_map.get(ts_day, 0.015) if yield_map else 0.015

                stats = process_day_statistics(last_valid, final_r, r_raw, q_val)
                if stats: results.append(stats)

    except KeyboardInterrupt:
        print("\nOprit de utilizator...")

    # Salvare
    if results:
        final_df = pd.DataFrame(results)
        final_df = final_df.sort_values('Date')
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nGenerare completa! Fisier salvat: {OUTPUT_FILE}")
        print(final_df[['Date', 'Risk_Free_Rate_Kalman', 'Risk_Free_Rate_Raw', 'Dividend_Yield_Continuous']].tail(3))