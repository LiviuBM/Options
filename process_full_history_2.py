import pandas as pd
import numpy as np
import py_vollib_vectorized
from scipy.stats import linregress
from tqdm import tqdm
import warnings
import os

# --- CONFIGURARE ---
FILE_PATH = 'options_eod_all.csv'
YIELD_FILE = 'S&P500Index-Returns.xlsx'  # <--- NUMELE CORECT ACUM
OUTPUT_FILE = 'SPX_Advanced_History_2012_2025.csv'
CHUNK_SIZE = 100000

# COLOANE EXACTE (Verificate)
COL_DATE = 'Calendar Date'
COL_RET_WITH_DIV = 'Value-Weighted Return-incl. dividends'
COL_RET_NO_DIV = 'Value-Weighted Return-excl. dividends'

warnings.filterwarnings('ignore')


# --- 1. FUNCTIE FLEXIBILA (XLSX sau CSV) ---
def load_dynamic_yields(yield_file_path):
    """
    Incarca datele indiferent daca e .csv sau .xlsx
    si calculeaza Dividend Yield Continuu (q).
    """
    print(f"--- Incarc datele de dividende din: {yield_file_path} ---")

    if not os.path.exists(yield_file_path):
        print(f"EROARE CRITICA: Fisierul {yield_file_path} nu exista in folder.")
        return None

    # Detectie extensie
    try:
        if yield_file_path.endswith('.xlsx'):
            # Citire EXCEL
            print("Format detectat: Excel (.xlsx)")
            df = pd.read_excel(yield_file_path, engine='openpyxl')
        else:
            # Citire CSV (cu fallback de encoding)
            print("Format detectat: CSV")
            try:
                df = pd.read_csv(yield_file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(yield_file_path, encoding='cp1252')

    except Exception as e:
        print(f"EROARE la citirea fisierului: {e}")
        return None

    # Curatam numele coloanelor (stergem spatii)
    df.columns = df.columns.str.strip()

    # Verificare coloane
    missing = [c for c in [COL_DATE, COL_RET_WITH_DIV, COL_RET_NO_DIV] if c not in df.columns]
    if missing:
        print(f"EROARE COLOANE LIPSA: {missing}")
        print(f"Coloane gasite: {list(df.columns)}")
        return None

    # Conversie Date
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df.sort_values(COL_DATE).set_index(COL_DATE)

    # Fortam numeric
    df[COL_RET_WITH_DIV] = pd.to_numeric(df[COL_RET_WITH_DIV], errors='coerce').fillna(0)
    df[COL_RET_NO_DIV] = pd.to_numeric(df[COL_RET_NO_DIV], errors='coerce').fillna(0)

    # Construire Indici Sintetici
    df['TR_Index'] = (1 + df[COL_RET_WITH_DIV]).cumprod()
    df['PR_Index'] = (1 + df[COL_RET_NO_DIV]).cumprod()

    # Calcul Yield (Trailing 252 zile)
    window = 252
    df['Ratio'] = df['TR_Index'] / df['PR_Index']

    # Yield Discret
    df['yield_discrete'] = (df['Ratio'] / df['Ratio'].shift(window)) - 1
    df['yield_discrete'] = df['yield_discrete'].fillna(method='bfill').interpolate()
    df['yield_discrete'] = df['yield_discrete'].clip(lower=0.0, upper=0.06)

    # Yield Continuu (CRITIC PENTRU BLACK-SCHOLES)
    df['yield_continuous'] = np.log(1 + df['yield_discrete'])

    # Mapare {Timestamp: float}
    yield_map = {k.normalize(): v for k, v in df['yield_continuous'].to_dict().items()}

    print(f"Succes! Yield continuu mediu: {df['yield_continuous'].mean():.2%}")
    return yield_map


# --- 2. RISK FREE RATE ---
def extract_implied_rf(day_df):
    if day_df.empty: return None
    S = day_df['underlying_bid_1545'].iloc[0]

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
            slope, _, _, _, _ = linregress(exp_data['strike'], exp_data['diff'])
            if -1.15 <= slope <= -0.85:
                r_implied = -np.log(-slope) / T
                if -0.02 <= r_implied < 0.15:
                    rates.append(r_implied)
        except:
            continue

    return np.median(rates) if rates else None


# --- 3. PROCESARE ---
def process_day_statistics(day_df, calculated_r, continuous_q):
    try:
        day_df['IV'] = py_vollib_vectorized.vectorized_implied_volatility(
            price=day_df['mid_price'].values,
            S=day_df['underlying_bid_1545'].values,
            K=day_df['strike'].values,
            t=day_df['T'].values,
            r=calculated_r,
            q=continuous_q,
            flag=day_df['flag'].values,
            return_as='numpy',
            on_error='ignore'
        )
    except Exception:
        return None

    day_df = day_df.dropna(subset=['IV'])
    day_df = day_df[(day_df['IV'] > 0.01) & (day_df['IV'] < 3.0)]

    if day_df.empty: return None
    S = day_df['underlying_bid_1545'].iloc[0]

    day_df['dist'] = (day_df['strike'] - S).abs()
    atm_opt = day_df.nsmallest(1, 'dist')
    atm_iv = atm_opt['IV'].values[0] if not atm_opt.empty else np.nan

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
        'Risk_Free_Rate': calculated_r,
        'Dividend_Yield_Continuous': continuous_q,
        'ATM_IV': atm_iv,
        'Skew_90_110': skew,
        'Kurtosis_Wings': kurt_proxy,
        'Options_Volume': len(day_df)
    }


# --- MAIN ---
if __name__ == "__main__":

    # 1. Incarcare Dividende (acum stie sa citeasca Excel)
    yield_map = load_dynamic_yields(YIELD_FILE)

    # Fallback values
    last_known_r = 0.015
    last_known_q = 0.020

    if yield_map is None:
        print("!!! ATENTIE: Rulez fara dividende dinamice (fisier incarcat gresit).")

    print(f"Start procesare optiuni din: {FILE_PATH}")
    reader = pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE)
    results = []
    buffer_df = pd.DataFrame()

    try:
        for chunk in tqdm(reader, desc="Procesare"):
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

            chunk['expiration'] = pd.to_datetime(chunk['expiration'])
            chunk['T'] = (chunk['expiration'] - chunk['quote_date']).dt.days / 365.0

            # Filtru Ask >= Bid (Integritate date)
            valid_prices = (chunk['ask_1545'] >= chunk['bid_1545'])
            chunk.loc[valid_prices, 'mid_price'] = (chunk.loc[valid_prices, 'bid_1545'] + chunk.loc[
                valid_prices, 'ask_1545']) / 2

            mask = valid_prices & (chunk['T'] > 0.005) & (chunk['mid_price'] > 0.05) & (
                        chunk['underlying_bid_1545'] > 0)
            valid_chunk = chunk[mask].copy()

            if valid_chunk.empty: continue
            valid_chunk['flag'] = valid_chunk['option_type'].str.lower().str.get(0)

            unique_days = valid_chunk['quote_date'].unique()
            for day in unique_days:
                day_data = valid_chunk[valid_chunk['quote_date'] == day].copy()

                r_calc = extract_implied_rf(day_data)
                final_r = (0.8 * last_known_r + 0.2 * r_calc) if r_calc is not None else last_known_r
                last_known_r = final_r

                ts_day = pd.to_datetime(day).normalize()
                q_val = yield_map.get(ts_day, last_known_q) if yield_map else last_known_q
                last_known_q = q_val

                stats = process_day_statistics(day_data, final_r, q_val)
                if stats: results.append(stats)

        # Buffer final
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
                r_calc = extract_implied_rf(last_valid)
                r_final = r_calc if r_calc is not None else last_known_r
                ts_day = last_valid['quote_date'].iloc[0].normalize()
                q_val = yield_map.get(ts_day, last_known_q) if yield_map else last_known_q
                stats = process_day_statistics(last_valid, r_final, q_val)
                if stats: results.append(stats)

    except KeyboardInterrupt:
        print("\nOprit de utilizator...")

    if results:
        final_df = pd.DataFrame(results)
        final_df = final_df.sort_values('Date')
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nGenerare completa! Fisier: {OUTPUT_FILE}")
        print(final_df[['Date', 'Risk_Free_Rate', 'Dividend_Yield_Continuous', 'ATM_IV']].tail(3))