import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import py_vollib_vectorized

# --- CONFIGURARE ---
FILE_PATH = 'options_eod_all.csv'  # <--- VERIFICĂ NUMELE FIȘIERULUI SĂ FIE CORECT (ex: options_2012.csv)
TARGET_DATE = '2012-01-03'
RISK_FREE = 0.04
DIVIDEND = 0.015


def get_data_for_date(file_path, target_date):
    """Citeste fisierul mare bucata cu bucata pana gaseste ziua tinta."""
    chunk_size = 100000  # Citim in bucati mai mici pentru siguranta

    # Verificam daca fisierul exista
    try:
        reader = pd.read_csv(file_path, chunksize=chunk_size)
    except FileNotFoundError:
        print(f"EROARE: Nu gasesc fisierul: {file_path}")
        return pd.DataFrame()

    found_data = []

    print(f"Caut date pentru {target_date}...")
    for chunk in reader:
        chunk['quote_date'] = pd.to_datetime(chunk['quote_date'])

        day_chunk = chunk[chunk['quote_date'] == target_date].copy()

        if not day_chunk.empty:
            found_data.append(day_chunk)

        if chunk['quote_date'].max() > pd.to_datetime(target_date):
            if not day_chunk.empty:
                continue
            break

    if found_data:
        return pd.concat(found_data)
    else:
        return pd.DataFrame()


# 1. Extragem datele
df = get_data_for_date(FILE_PATH, TARGET_DATE)

if df.empty:
    print("Nu s-au gasit date pentru aceasta zi sau fisierul nu a fost gasit.")
else:
    print(f"Date brute gasite: {len(df)} optiuni.")

    # 2. PROCESARE SI CURATARE AGRESIVA
    df = df.copy()

    # Conversie date
    df['expiration'] = pd.to_datetime(df['expiration'])
    df['T'] = (df['expiration'] - df['quote_date']).dt.days / 365.0

    # Calcul Mid Price
    df['mid_price'] = (df['bid_1545'] + df['ask_1545']) / 2

    # --- FILTRE DE SIGURANTA (CRITIC PENTRU A EVITA ZeroDivisionError) ---
    original_len = len(df)

    mask = (
            (df['T'] > 0.005) &  # Timp > 2 zile (evitam diviziunea la 0 timp)
            (df['strike'] > 1) &  # Strike valid
            (df['underlying_bid_1545'] > 1) &  # Underlying valid
            (df['mid_price'] > 0.05) &  # Pret optiune valid (fara penny options moarte)
            (df['mid_price'].notna())  # Fara NaN
    )
    df = df[mask].copy()

    print(f"Au ramas {len(df)} optiuni valide dupa filtrare (eliminați {original_len - len(df)} garbage data).")

    if df.empty:
        print("Toate datele au fost filtrate. Verifica daca data aleasa are date valide.")
        exit()

    # Pregatire flag
    df['flag'] = df['option_type'].str.lower().str.get(0)

    print("Calculez IV (asta poate dura cateva secunde)...")

    try:
        # Folosim .values pentru a fi siguri ca trimitem numpy arrays curate, fara indexi pandas
        iv_results = py_vollib_vectorized.vectorized_implied_volatility(
            price=df['mid_price'].values,
            S=df['underlying_bid_1545'].values,
            K=df['strike'].values,
            t=df['T'].values,
            r=RISK_FREE,
            q=DIVIDEND,
            flag=df['flag'].values,
            return_as='numpy',
            on_error='ignore'  # Returneaza NaN in loc sa crape
        )

        df['IV'] = iv_results

    except Exception as e:
        print(f"EROARE FATALA la calculul IV: {e}")
        # Debugging: afisam valorile minime care ar fi putut cauza eroarea
        print(f"Min T: {df['T'].min()}, Min K: {df['strike'].min()}, Min S: {df['underlying_bid_1545'].min()}")
        exit()

    # Eliminam rezultatele unde IV nu s-a putut calcula (NaN)
    df = df.dropna(subset=['IV'])

    # Filtram IV-uri nerealiste (de ex: > 200% sau < 1% pentru SPX e zgomot)
    df = df[(df['IV'] > 0.01) & (df['IV'] < 2.0)]

    print(f"IV calculat cu succes pentru {len(df)} optiuni. Generez graficul...")

    # 3. PLOTARE 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    X = df['strike']
    Y = df['T']
    Z = df['IV']

    # Plotare Scatter 3D (mai sigur decat surface daca datele nu sunt uniforme)
    # Folosim culori bazate pe IV
    scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=10, alpha=0.8)

    ax.set_title(f'Volatility Surface SPX - {TARGET_DATE}')
    ax.set_xlabel('Strike ($)')
    ax.set_ylabel('Time to Maturity (Years)')
    ax.set_zlabel('Implied Volatility')

    # Adaugam colorbar
    plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1, label='Implied Volatility')

    plt.show()