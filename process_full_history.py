import pandas as pd
import numpy as np
import py_vollib_vectorized
from scipy.stats import linregress
from tqdm import tqdm
import warnings

# --- CONFIGURARE ---
FILE_PATH = 'options_eod_all.csv'  # <--- MODIFICĂ AICI
OUTPUT_FILE = 'SPX_Advanced_History_Refined.csv'
CHUNK_SIZE = 100000  # Procesare pe bucăți
DIVIDEND_YIELD = 0.015  # Dividend Yield mediu (SPX)

# Ignorăm avertismentele de calcul (ex: impartire la zero in cazuri izolate)
warnings.filterwarnings('ignore')


def extract_implied_rf(day_df):
    """
    Calculează Rata fără Risc (r) folosind Put-Call Parity Regression.
    Formula: Call - Put = S*e^(-qT) - K*e^(-rT)
    Regresie Liniară: Y(C-P) vs X(Strike).
    Panta (Slope) = -e^(-rT).
    Deci: r = -ln(-Slope) / T
    """
    if day_df.empty:
        return 0.04  # Fallback

    # Filtrare: Doar optiuni "At The Money" (+/- 20%) și cu T > 1 săptămână
    # Acestea au cea mai bună lichiditate și respectă cel mai bine paritatea
    S = day_df['underlying_bid_1545'].iloc[0]
    subset = day_df[
        (day_df['strike'] > S * 0.80) &
        (day_df['strike'] < S * 1.20) &
        (day_df['T'] > 0.02)
        ].copy()

    if subset.empty:
        return 0.04

        # Separam Calls si Puts
    calls = subset[subset['flag'] == 'c'][['expiration', 'strike', 'mid_price', 'T']]
    puts = subset[subset['flag'] == 'p'][['expiration', 'strike', 'mid_price']]

    # Unim pe Expirare și Strike pentru a găsi perechile
    merged = pd.merge(calls, puts, on=['expiration', 'strike'], suffixes=('_c', '_p'))

    if len(merged) < 5:
        return 0.04

    merged['diff'] = merged['mid_price_c'] - merged['mid_price_p']
    rates = []

    # Calculăm r pentru fiecare dată de expirare disponibilă în acea zi
    for exp in merged['expiration'].unique():
        exp_data = merged[merged['expiration'] == exp]

        # Avem nevoie de minim 4 puncte pentru o regresie cat de cat ok
        if len(exp_data) < 4:
            continue

        T = exp_data['T'].iloc[0]

        # Regresie: Diff ~ Strike
        # Panta ar trebui să fie negativă
        try:
            slope, intercept, r_val, p_val, std_err = linregress(exp_data['strike'], exp_data['diff'])

            # Verificăm dacă panta are sens (trebuie să fie aprox -1)
            # Acceptăm toleranță între -1.1 și -0.9
            if -1.1 <= slope <= -0.9:
                r_implied = -np.log(-slope) / T

                # Filtru de bun simț: Ratele trebuie să fie între 0% și 15%
                if 0.0 <= r_implied < 0.15:
                    rates.append(r_implied)
        except:
            continue

    # Returnăm mediana ratelor (pentru a elimina outlierii)
    if rates:
        return np.median(rates)
    else:
        return 0.04  # Fallback default dacă regresia eșuează


def process_day_statistics(day_df, calculated_r):
    """
    Calculează IV cu rata specifică și extrage statistici.
    """
    # 1. Calcul IV Vectorizat folosind Rata Dinamică
    try:
        day_df['IV'] = py_vollib_vectorized.vectorized_implied_volatility(
            price=day_df['mid_price'].values,
            S=day_df['underlying_bid_1545'].values,
            K=day_df['strike'].values,
            t=day_df['T'].values,
            r=calculated_r,  # <--- FOLOSIM RATA CALCULATA DINAMIC
            q=DIVIDEND_YIELD,
            flag=day_df['flag'].values,
            return_as='numpy',
            on_error='ignore'
        )
    except:
        return None

    # Eliminam erori
    day_df = day_df.dropna(subset=['IV'])
    # Filtram IV-uri extreme
    day_df = day_df[(day_df['IV'] > 0.01) & (day_df['IV'] < 3.0)]

    if day_df.empty:
        return None

    S = day_df['underlying_bid_1545'].iloc[0]

    # 2. ATM IV
    day_df['dist'] = (day_df['strike'] - S).abs()
    atm_opt = day_df.nsmallest(1, 'dist')
    atm_iv = atm_opt['IV'].values[0] if not atm_opt.empty else np.nan

    # 3. Skew & Kurtosis
    # Definim zonele de interes (Wings)
    put_90 = day_df[(day_df['strike'] >= S * 0.88) & (day_df['strike'] <= S * 0.92) & (day_df['flag'] == 'p')]
    call_110 = day_df[(day_df['strike'] >= S * 1.08) & (day_df['strike'] <= S * 1.12) & (day_df['flag'] == 'c')]

    put_80 = day_df[(day_df['strike'] >= S * 0.78) & (day_df['strike'] <= S * 0.82) & (day_df['flag'] == 'p')]
    call_120 = day_df[(day_df['strike'] >= S * 1.18) & (day_df['strike'] <= S * 1.22) & (day_df['flag'] == 'c')]

    iv_p90 = put_90['IV'].mean()
    iv_c110 = call_110['IV'].mean()
    iv_p80 = put_80['IV'].mean()
    iv_c120 = call_120['IV'].mean()

    # Calcul final metrici
    skew = iv_p90 - iv_c110
    kurt_proxy = ((iv_p80 + iv_c120) / 2) - atm_iv

    return {
        'Date': day_df['quote_date'].iloc[0],
        'Underlying_Price': S,
        'Risk_Free_Rate': calculated_r,  # Salvam rata calculata pentru verificare
        'ATM_IV': atm_iv,
        'Skew_90_110': skew,
        'Kurtosis_Wings': kurt_proxy,
        'Options_Volume': len(day_df)
    }


# --- MAIN EXECUTION ---

print(f"Start procesare: {FILE_PATH}")
print("Se calculeaza Rata Fara Risc dinamic din Put-Call Parity...")

reader = pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE)
results = []
buffer_df = pd.DataFrame()

try:
    for chunk in tqdm(reader, desc="Procesare Chunks"):

        # 1. Pregătire Chunk
        chunk['quote_date'] = pd.to_datetime(chunk['quote_date'])

        # Gestionare Buffer (lipim resturile de la chunk-ul anterior)
        if not buffer_df.empty:
            chunk = pd.concat([buffer_df, chunk])
            buffer_df = pd.DataFrame()

        # Verificăm dacă ultima zi din chunk este completă
        last_day = chunk['quote_date'].max()
        if chunk['quote_date'].nunique() > 1:
            # Salvăm ultima zi în buffer pentru următorul ciclu
            mask_last = chunk['quote_date'] == last_day
            buffer_df = chunk[mask_last].copy()
            chunk = chunk[~mask_last].copy()
        else:
            # Dacă chunk-ul e doar o parte dintr-o zi imensă, o punem în buffer
            buffer_df = pd.concat([buffer_df, chunk])
            continue

        # 2. Curățare Date (Filtre de Siguranță)
        chunk['expiration'] = pd.to_datetime(chunk['expiration'])
        chunk['T'] = (chunk['expiration'] - chunk['quote_date']).dt.days / 365.0
        chunk['mid_price'] = (chunk['bid_1545'] + chunk['ask_1545']) / 2

        mask = (
                (chunk['T'] > 0.005) &
                (chunk['strike'] > 1) &
                (chunk['underlying_bid_1545'] > 1) &
                (chunk['mid_price'] > 0.05)
        )
        valid_chunk = chunk[mask].copy()

        if valid_chunk.empty:
            continue

        valid_chunk['flag'] = valid_chunk['option_type'].str.lower().str.get(0)

        # 3. Iterare pe Zilele din Chunk
        unique_days = valid_chunk['quote_date'].unique()

        for day in unique_days:
            day_data = valid_chunk[valid_chunk['quote_date'] == day].copy()

            # A. Calculam Rata Dobanzii (Implied Rate)
            implied_r = extract_implied_rf(day_data)

            # B. Calculam Statistici cu rata gasita
            stats = process_day_statistics(day_data, implied_r)

            if stats:
                results.append(stats)

    # Procesare finala buffer (ultima zi din fisier)
    if not buffer_df.empty:
        # Repetam logica pentru ultimul rest
        buffer_df['expiration'] = pd.to_datetime(buffer_df['expiration'])
        buffer_df['T'] = (buffer_df['expiration'] - buffer_df['quote_date']).dt.days / 365.0
        buffer_df['mid_price'] = (buffer_df['bid_1545'] + buffer_df['ask_1545']) / 2
        mask = (buffer_df['T'] > 0.005) & (buffer_df['mid_price'] > 0.05)
        last_valid = buffer_df[mask].copy()

        if not last_valid.empty:
            last_valid['flag'] = last_valid['option_type'].str.lower().str.get(0)
            r = extract_implied_rf(last_valid)
            stats = process_day_statistics(last_valid, r)
            if stats:
                results.append(stats)

except KeyboardInterrupt:
    print("\nProcesare oprita de utilizator. Salvez ce am gasit pana acum...")

# Salvare Rezultate
if results:
    final_df = pd.DataFrame(results)
    final_df = final_df.sort_values('Date')
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSucces! Fisier generat: {OUTPUT_FILE}")
    print(final_df.head())
else:
    print("\nNu s-au generat rezultate. Verifica filtrele sau fisierul.")