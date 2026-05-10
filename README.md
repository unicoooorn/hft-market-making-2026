# HFT Market Making Backtester

A for-loop backtesting system for analyzing HFT market making strategies.
   
## Usage

### Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip sync pyproject.toml
```

### Run Backtest
```bash
PYTHONPATH=src python main.py
```

### Grid Search
```bash
PYTHONPATH=src python scripts/grid_search.py
```

### Analyze Data
```bash
PYTHONPATH=src python scripts/analyze_lob.py
PYTHONPATH=src python scripts/analyze_trades.py
```

### Convert CSV to Parquet
```bash
PYTHONPATH=src python scripts/convert_to_parquet.py
```

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

48 tests covering:
- Inventory tracking
- AS strategy (volatility, quotes, cancellation)
- Data provider streaming
- Chart generation
