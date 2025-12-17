# SPX Options Analytics & Volatility Surface Pipeline

## Scop general

Repozitoriul implementează un pipeline complet de analiză a opțiunilor pe S&P 500 (SPX), pornind de la date brute EOD, extrăgând parametri implicați (volatilitate, rată fără risc, dividend), construind serii istorice robuste și culminând cu vizualizări avansate: dashboard analitic și animație 3D a suprafeței de volatilitate.

Pipeline-ul este gândit pentru **cercetare cantitativă**, nu pentru trading live: accent pe stabilitate statistică, filtrare a zgomotului și interpretabilitate economică.

---

## Structura repo-ului

```
.
├── options_eod_all.csv
├── S&P500Index-Returns.xlsx
├── SPX_Advanced_History_Refined.csv   (generat)
├── plot_volatility_surface_v3.py
├── plot_dashboard_v3.py
├── video_generator.py
├── image_generator_v3.py
└── frames_output/                     (generat)
```

---

## Dependințe

* Python >= 3.9
* pandas
* numpy
* scipy
* matplotlib
* seaborn
* py_vollib_vectorized
* tqdm
* openpyxl
* opencv-python

Instalare rapidă:

```bash
pip install pandas numpy scipy matplotlib seaborn py_vollib_vectorized tqdm openpyxl opencv-python
```

---

## 1. `plot_volatility_surface_v3.py`

### Rol

Scriptul **central** de procesare statistică. Transformă datele brute de opțiuni într-o serie zilnică de parametri implicați, stabili și coerent definiți economic.

### Input

* `options_eod_all.csv` – opțiuni SPX EOD (bid/ask, strike, maturitate, tip)
* `S&P500Index-Returns.xlsx` – randamente Total Return vs Price Return

### Output

* `SPX_Advanced_History_Refined.csv`

### Ce calculează

Pentru fiecare zi:

* **Rata fără risc implicită**

  * estimare brută din Put–Call Parity
  * filtrare cu **Kalman Filter 1D (random walk)**
* **Dividend yield continuu**

  * derivat structural din raport TR / PR
  * trailing window 252 zile
* **Volatilitate implicită ATM**

  * mediană robustă într-o bandă ±3% în jurul spot
* **Skew**: IV(put 90%) − IV(call 110%)
* **Proxy de kurtosis**: wing-uri 80% / 120% vs ATM

### Elemente cheie de design

* Chunking pentru fișiere foarte mari
* Buffer pe zile incomplete
* Filtrare agresivă a datelor eronate (bid/ask, T, preț)
* Vectorizare completă pentru IV

### Rulare

```bash
python plot_volatility_surface_v3.py
```

---

## 2. `plot_dashboard_v3.py`

### Rol

Vizualizare sintetică, **interpretabilă macro**, a seriilor istorice extrase.

### Input

* `SPX_Advanced_History_Refined.csv`

### Output

* Dashboard Matplotlib (4 panouri)

### Panouri

1. **Market Fear Gauge**

   * SPX Price vs ATM IV (SMA)
2. **Option Skew**

   * cerere relativă de protecție (puts vs calls)
3. **Risk-Free Rate**

   * Raw vs Kalman (zgomot vs semnal)
4. **Dividend Yield continuu**

### Observații

* SMA folosit doar pentru lizibilitate
* Ratele și yield-ul sunt deja filtrate structural

### Rulare

```bash
python plot_dashboard_v3.py
```

---

## 3. `image_generator_v3.py`

### Rol

Generează **frame-uri zilnice 3D** ale suprafeței de volatilitate, consistente în timp, pentru animație.

### Input

* `options_eod_all.csv`
* `SPX_Advanced_History_Refined.csv`

### Output

* `frames_output/frame_00000.png`, `frame_00001.png`, ...

### Caracteristici critice

* Axe **fixe** (moneyness, maturitate, IV)
* Moneyness definit forward: `K / F`
* Doar opțiuni OTM
* Filtre anti-spike:

  * spread contextual (ATM vs wings)
  * quantile clipping
  * interpolare + Gaussian smoothing

### Motivare

Asigură că variațiile vizuale reflectă **schimbări reale de structură**, nu artefacte de date.

### Rulare

```bash
python image_generator_v3.py
```

---

## 4. `video_generator.py`

### Rol

Convertește frame-urile generate într-un video MP4.

### Input

* `frames_output/*.png`

### Output

* `SPX_Volatility_Evolution.mp4`

### Detalii

* Sortare alfanumerică strictă (frame_00001, ...)
* Codec standard `mp4v`
* FPS configurabil (default 24)

### Rulare

```bash
python video_generator.py
```

---

## Ordine recomandată de execuție

1. `plot_volatility_surface_v3.py`
2. `plot_dashboard_v3.py`
3. `image_generator_v3.py`
4. `video_generator.py`

---

## Note metodologice

* Rata fără risc este **endogenă** (extrasă din opțiuni), nu impusă extern
* Dividendul este **structural**, nu estimat din opțiuni (evită instabilitatea)
* Kalman Filter este folosit exclusiv unde ipoteza de random walk este economic justificată

---

## Limitări cunoscute

* Nu tratează zile cu lichiditate extrem de redusă
* Nu include ajustări pentru rate negative extreme
* Nu este optimizat pentru intraday

