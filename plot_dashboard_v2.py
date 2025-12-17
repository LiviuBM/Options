import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# --- CONFIGURARE ---
FILE_PATH = 'SPX_Advanced_History_2012_2025.csv'


def plot_dashboard():
    # 1. Incarcare date
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Nu gasesc fisierul {FILE_PATH}. Verifica daca l-ai generat.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # 2. Smoothing (Medii mobile) pentru vizibilitate
    # Folosim o fereastra de 5 zile pentru a elimina zgomotul zilnic
    df['IV_SMA'] = df['ATM_IV'].rolling(5).mean() * 100  # Procentual
    df['Skew_SMA'] = df['Skew_90_110'].rolling(10).mean()
    df['R_SMA'] = df['Risk_Free_Rate'].rolling(20).mean() * 100
    df['Q_SMA'] = df['Dividend_Yield_Continuous'].rolling(20).mean() * 100

    # Setare stil vizual
    sns.set_theme(style="darkgrid")

    # 3. Creare Panou Grafic (4 rânduri)
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

    # --- A. ATM Volatility & Price ---
    ax1 = axes[0]
    ax1_right = ax1.twinx()

    # Pretul SPX (Linie subtire gri)
    ln1 = ax1_right.plot(df['Date'], df['Underlying_Price'], color='grey', alpha=0.3, label='SPX Price (Right)')
    # Volatilitatea (Linie principala)
    ln2 = ax1.plot(df['Date'], df['IV_SMA'], color='#d62728', linewidth=1.5, label='ATM Implied Volatility (%)')

    ax1.set_title('Market Fear Gauge: SPX Price vs. Implied Volatility', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Implied Volatility (%)', color='#d62728', fontweight='bold')
    ax1_right.set_ylabel('SPX Price', color='grey')

    # Legenda combinata
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
    ax2.text(df['Date'].iloc[0], df['Skew_SMA'].max() * 0.9, "High Skew = High Demand for Crash Protection", fontsize=9,
             bbox=dict(facecolor='white', alpha=0.7))

    # --- C. Risk Free Rate (Derived) ---
    ax3 = axes[2]
    ax3.plot(df['Date'], df['R_SMA'], color='#2ca02c', linewidth=1.5)
    ax3.set_title('Implied Risk-Free Rate (via Put-Call Parity)', fontsize=12)
    ax3.set_ylabel('Interest Rate (%)')

    # --- D. Dividend Yield (Continuous) ---
    ax4 = axes[3]
    ax4.plot(df['Date'], df['Q_SMA'], color='#9467bd', linewidth=1.5)
    ax4.set_title('Implied Continuous Dividend Yield', fontsize=12)
    ax4.set_ylabel('Yield (%)')
    ax4.set_xlabel('Date')

    # Formatare axa X
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_dashboard()