# SPX Options Analytics & Volatility Surface Pipeline

## General Purpose

This repository implements a full analytics pipeline for S&P 500 (SPX) options, starting from raw end-of-day (EOD) option data and extracting endogenous implied parameters (volatility, risk-free rate, dividend yield). The pipeline constructs robust historical time series and produces advanced visualizations: a macro-analytic dashboard and a 3D volatility surface animation.

The pipeline is designed for **quantitative research**, not live trading. Emphasis is placed on statistical stability, structural noise filtering, and economic interpretability.

---

## Repository Structure

.
├── options_eod_all.csv  
├── S&P500Index-Returns.xlsx  
├── SPX_Advanced_History_Refined_v2.csv  
├── plot_volatility_surface_v4.py  
├── plot_dashboard_v3.py  
├── image_generator_v4.py  
├── video_generator.py  
└── frames_output_v2/  

---

## Dependencies

- Python >= 3.9  
- pandas  
- numpy  
- scipy  
- matplotlib  
- py_vollib_vectorized  
- tqdm  
- openpyxl  
- opencv-python  

### Quick Installation

    pip install pandas numpy scipy matplotlib py_vollib_vectorized tqdm openpyxl opencv-python

---

## 1. plot_volatility_surface_v4.py

### Role

Core statistical processing script. Transforms raw option data into a daily time series of implied parameters that are temporally stable and economically well-defined.

### Input

- options_eod_all.csv – SPX EOD options (bid/ask, strike, maturity, type)  
- S&P500Index-Returns.xlsx – Value-Weighted returns with and without dividends  

### Output

- SPX_Advanced_History_Refined_v2.csv  

### Computed Metrics (per day)

Risk-Free Rate (implied)
- raw estimation via Put–Call Parity (strike–spread regression)
- filtering using 1D Kalman Filter (random walk)
- explicit separation of Raw vs Kalman (noise vs signal)

Continuous Dividend Yield
- structurally derived from Total Return / Price Return ratio
- trailing window: 252 days
- converted to continuous yield for Black–Scholes

ATM Implied Volatility
- robust median within a ±3% band around spot

Skew
- IV(put 90%) − IV(call 110%)

Kurtosis Proxy
- 80% / 120% wings relative to ATM

### Key Design Choices

- chunking for very large datasets
- explicit buffering for incomplete trading days
- literature-validated filters (bid/ask, intrinsic bounds, minimum maturity)
- aggressive cleaning of erroneous observations
- fully vectorized IV computation
- Kalman Filter applied exclusively to the risk-free rate

### Execution

    python plot_volatility_surface_v4.py

---

## 2. plot_dashboard_v3.py

### Role

Macro-level, interpretable visualization of the historical series extracted by the pipeline.

### Input

- SPX_Advanced_History_Refined_v2.csv  

### Output

- Matplotlib dashboard (4 panels)

### Panels

Market Fear Gauge
- SPX price vs ATM IV (SMA used only for visual clarity)

Option Skew
- proxy for relative demand for downside protection (puts vs calls)

Risk-Free Rate
- Raw vs Kalman (noise vs signal)

Continuous Dividend Yield
- structural series, not option-implied

### Notes

- SMA is used strictly for visualization clarity
- rates and yields are already structurally filtered

### Execution

    python plot_dashboard_v3.py

---

## 3. image_generator_v4.py

### Role

Generates daily 3D frames of the volatility surface, consistent through time, used for animation.

### Input

- options_eod_all.csv  
- SPX_Advanced_History_Refined_v2.csv  

### Output

- frames_output_v2/frame_00000.png  
- frames_output_v2/frame_00001.png  
- ...

### Critical Characteristics

- fixed axes: moneyness, maturity, IV
- forward-defined moneyness: K / F
- forward price explicitly computed: F = S · exp((r − q)T)
- OTM options only

### Anti-Spike Filters (v4)

- contextual spread filter (strict at ATM, permissive in wings)
- quantile clipping (lower 2%)
- axis-bound filtering
- linear + nearest interpolation
- controlled Gaussian smoothing (σ = 0.8)

### Motivation

Ensures that visual variations reflect genuine structural changes in the volatility surface rather than data artifacts.

### Execution

    python image_generator_v4.py

---

## 4. video_generator.py

### Role

Converts generated frames into an MP4 video.

### Input

- frames_output_v2/*.png  

### Output

- SPX_Volatility_Evolution.mp4  

### Details

- strict alphanumeric sorting
- standard mp4v codec
- configurable FPS (default: 24)

### Execution

    python video_generator.py

---

## Recommended Execution Order

1. plot_volatility_surface_v4.py  
2. plot_dashboard_v3.py  
3. image_generator_v4.py  
4. video_generator.py  

---

## Methodological Notes

- the risk-free rate is endogenous (extracted from options), not externally imposed
- dividends are structural, derived from TR/PR indices, not option-implied
- Kalman Filter is used only where the random walk assumption is economically justified
- the surface is built forward-consistently (K/F), not spot-based

---

## Known Limitations

- does not explicitly handle days with extremely low liquidity
- no adjustments for extreme negative rate regimes
- not optimized for intraday data
