# Volatility modeling in financial markets

## Overview
This project studies short-term volatility dynamics in financial time series using statistical features and ML models.

The objective is to predict future volatility from past market behavior.

The pipeline is asset-agnostic, and can be applied to equities, commodities or other financial instruments. The current implementation uses WTI crude oil as case study.

## Methodology

### Data
- Daily price data (from Yahoo finance)
- Asset (WTI crude oil: `CL=F`)
- Long historical period (>10 years)

### Preprocessing
- Log returns are computed from the time series
- Data is aligned and made consistent with the time order

### Features
- Lagged returns (`1, 2, 5, 10`)
- Absolute value of lag-1 return (`|r_(t-1)|`)
- Rolling statistics:
  - Mean (5 days and 20 days)
  - Volatility (5 days and 20 days)

### Target
- Future volatility: rolling standard deviation over the next 5 days (shifted to avoid look-ahead bias)

### Models
- Baseline: volatility persistence
- Ridge regression: linear model
- Random forest: nonlinear model

### Evaluation
- Chronological train/test split (no shuffling)
- Walk-forward evaluation with expanding training window
- Metrics
  - Mean absolute error (MAE)
  - Mean squared error (MSE)

### Key results
Results are evaluated on a held-out test set and validated using walk-forward evaluation.

- The baseline model (volatility persistence) already provides strong predictive performance
- Ridge and Random Forest models achieve modest but consistent improvements
- Random Forest slightly outperforms linear models, indicating weak nonlinear effects
- Volatility is primarily driven by its own past (volatility clustering)
- Long-term volatility (20-day window) is the most important feature
- Short-term volatility and recent trends contribute additional signal
- Lagged returns have negligible predictive power
- All the models tend to underestimate the extreme volatility spikes
- Walk-forward evaluation shows higher errors than a single split but confirms that model improvements are stable across time

## How to run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run pipeline

Running `python main.py` saves predictions, metrics, and trained models in the `results/` folder.

### Analyze results

Open `notebooks/02_visualization.ipynb` to visualize predictions and feature importance.