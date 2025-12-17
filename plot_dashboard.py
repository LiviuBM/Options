import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- CONFIGURARE ---
FILE_PATH = 'SPX_Advanced_History_2012_2025.csv'

# Citim datele
df = pd.read_csv(FILE_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Aplicam o medie mobila scurta (5 zile) pentru a netezi zgomotul
df['ATM_IV_SMA'] = df['ATM_IV'].rolling(window=5).mean()
df['Skew_SMA'] = df['Skew_90_110'].rolling(window=5).mean()
df['RF_SMA'] = df['Risk_Free_Rate'].rolling(window=10).mean()

# --- GENERARE GRAFICE ---
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

# 1. SPX Price vs ATM Volatility
ax1 = axes[0]
ax1.set_title('SPX Price vs Implied Volatility (ATM)', fontsize=14, fontweight='bold')
ax1.plot(df['Date'], df['Underlying_Price'], color='black', label='SPX Price', linewidth=1)
ax1.set_ylabel('SPX Price', color='black')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Axa secundara pentru IV
ax1_bis = ax1.twinx()
ax1_bis.plot(df['Date'], df['ATM_IV_SMA'], color='red', alpha=0.6, label='ATM IV (Rolling)', linewidth=1)
ax1_bis.set_ylabel('Implied Volatility', color='red')
ax1_bis.fill_between(df['Date'], df['ATM_IV_SMA'], 0, color='red', alpha=0.1)
ax1_bis.legend(loc='upper right')

# 2. Skew (Fear Gauge)
ax2 = axes[1]
ax2.set_title('Volatility Skew (Put IV - Call IV)', fontsize=12)
ax2.plot(df['Date'], df['Skew_SMA'], color='blue', linewidth=1)
ax2.axhline(0, color='black', linestyle='--')
ax2.set_ylabel('Skew Magnitude')
ax2.fill_between(df['Date'], df['Skew_SMA'], df['Skew_SMA'].mean(), where=(df['Skew_SMA'] > df['Skew_SMA'].mean()), color='blue', alpha=0.1)
ax2.grid(True, alpha=0.3)
ax2.text(df['Date'].iloc[0], df['Skew_SMA'].max(), 'High Skew = High Demand for Puts (Fear)', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

# 3. Kurtosis (Tail Risk)
ax3 = axes[2]
ax3.set_title('Kurtosis Proxy (Tail Risk / Fat Tails)', fontsize=12)
ax3.plot(df['Date'], df['Kurtosis_Wings'].rolling(5).mean(), color='purple', linewidth=1)
ax3.set_ylabel('Kurtosis Proxy')
ax3.grid(True, alpha=0.3)

# 4. Implied Risk-Free Rate (Put-Call Parity)
ax4 = axes[3]
ax4.set_title('Implied Risk-Free Rate (Derived from Options)', fontsize=12)
ax4.plot(df['Date'], df['RF_SMA'], color='green', linewidth=1.5)
ax4.set_ylabel('Rate (%)')
ax4.grid(True, alpha=0.3)

# Formatare axa X
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# --- ANALIZA SIMPLA TEXT ---
print(f"Statistici Generale 2012-2025:")
print(f"Min ATM IV: {df['ATM_IV'].min():.2%}")
print(f"Max ATM IV: {df['ATM_IV'].max():.2%} (Probabil Martie 2020)")
print(f"Media Skew: {df['Skew_90_110'].mean():.4f}")
print(f"Max Rate detectata: {df['Risk_Free_Rate'].max():.2%}")