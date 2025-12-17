import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# --- CONFIGURARE ---
# Acum citim fisierul generat de scriptul Kalman V2
FILE_PATH = 'SPX_Advanced_History_Refined.csv'


def plot_dashboard():
    # 1. Incarcare date
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Nu gasesc fisierul {FILE_PATH}. Ruleaza intai procesarea istoricului.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # 2. Smoothing pentru vizualizare (IV si Skew)
    # Nota: Rata si Dividendul sunt deja filtrate Kalman, nu au nevoie de SMA
    df['IV_SMA'] = df['ATM_IV'].rolling(5).mean() * 100
    df['Skew_SMA'] = df['Skew_90_110'].rolling(10).mean()

    # Conversie procentuala pentru afisare
    df['R_Kalman'] = df['Risk_Free_Rate_Kalman'] * 100
    df['R_Raw'] = df['Risk_Free_Rate_Raw'] * 100
    df['Q_Continuous'] = df['Dividend_Yield_Continuous'] * 100

    # Setare stil vizual
    sns.set_theme(style="darkgrid")

    # 3. Creare Panou Grafic (4 rânduri)
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

    # --- A. Market Fear Gauge: Price vs Volatility ---
    ax1 = axes[0]
    ax1_right = ax1.twinx()

    ln1 = ax1_right.plot(df['Date'], df['Underlying_Price'], color='grey', alpha=0.3, label='SPX Price')
    ln2 = ax1.plot(df['Date'], df['IV_SMA'], color='#d62728', linewidth=1.5, label='ATM IV (%)')

    ax1.set_title('Market Fear Gauge: SPX Price vs. Implied Volatility', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Implied Volatility (%)', color='#d62728', fontweight='bold')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')

    # --- B. Skew (Put vs Call Demand) ---
    ax2 = axes[1]
    ax2.plot(df['Date'], df['Skew_SMA'], color='#1f77b4', linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Option Skew (90% Put IV - 110% Call IV)', fontsize=12)
    ax2.set_ylabel('Skew Magnitude')
    ax2.fill_between(df['Date'], df['Skew_SMA'], 0, where=(df['Skew_SMA'] > 0), color='#1f77b4', alpha=0.1)

    # --- C. Risk Free Rate (RAW vs KALMAN) ---
    ax3 = axes[2]
    # Plotam Raw (zgomot) in spate
    ax3.plot(df['Date'], df['R_Raw'], color='grey', alpha=0.4, linewidth=1, label='Raw Put-Call Parity')
    # Plotam Kalman (semnal) peste
    ax3.plot(df['Date'], df['R_Kalman'], color='#2ca02c', linewidth=2, label='Kalman Filter Estimate')

    ax3.set_title('Risk-Free Rate: Noise (Raw) vs Signal (Kalman)', fontsize=12)
    ax3.set_ylabel('Interest Rate (%)')
    ax3.legend(loc='upper left')

    # --- D. Dividend Yield (Continuous) ---
    ax4 = axes[3]
    ax4.plot(df['Date'], df['Q_Continuous'], color='#9467bd', linewidth=1.5)
    ax4.set_title('Implied Continuous Dividend Yield (TR vs PR)', fontsize=12)
    ax4.set_ylabel('Yield (%)')
    ax4.set_xlabel('Date')

    # Formatare axa X
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_dashboard()