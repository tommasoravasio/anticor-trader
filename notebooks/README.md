# Notebooks

This directory contains the core analysis and backtesting notebooks for the anticor-trader project.

## Structure

### 📋 `00_algorithm_template.ipynb`
**The core algorithm template** - All backtests are built on this template. Contains:
- ANTI² strategy implementation
- Data fetching from WRDS
- Performance calculation and visualization
- Results persistence

Start here to understand the algorithm.

### 📊 `backtests/`
**Portfolio backtest notebooks** - Each notebook runs the ANTI² algorithm on a different portfolio and saves results to `results/backtests/`.

- `01_sp500_top25.ipynb` - S&P 500 Top 25 stocks
- `02_sp500_25diversified.ipynb` - S&P 500 25 stocks (diversified)
- `03_nyse_top25.ipynb` - NYSE Top 25 stocks
- `04_ma25.ipynb` - 25 Moving Average stocks
- `05_etf25.ipynb` - 25 ETFs
- `06_maximumfactors.ipynb` - 25 Maximum Factors stocks
- `07_riskdiversified15.ipynb` - 15 Risk-Diversified stocks

**To run:** Execute cells sequentially. Each notebook:
1. Fetches data for its portfolio
2. Runs the ANTI² algorithm
3. Generates performance metrics and charts
4. Saves results to `../results/backtests/`

### 📈 `analysis/`
**Analysis and comparison notebooks** - Analyzes and compares results across tests.

- `10_synthetic_analysis.ipynb` - Analyzes synthetic market scenarios from `src/synthetic_data.py`

## Workflow

1. **Start with** `00_algorithm_template.ipynb` - understand the algorithm
2. **Run any backtest** in `backtests/` - generates results for a specific portfolio
3. **Analyze results** in `analysis/` - compare across portfolios and scenarios

## Data & Results

- **Input data**: `../data/` - Stock prices and metadata
- **Results**: `../results/backtests/` - CSV and JSON results from backtest notebooks
- **Synthetic results**: `../results/synthetic/` - Output from `src/synthetic_data.py`

## Dependencies

- Python 3.x with pandas, numpy, matplotlib
- WRDS credentials in `../wrds_credentials.txt` (for data fetching)
- `src.persist_backtest` module for saving results

