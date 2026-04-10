from src.config import load_config
from src.data import download_data, compute_log_returns
from src.features import build_features, build_target
from src.model import train_and_evaluate, walk_forward_evaluation
import json
import joblib

def main():
    config  = load_config("./config/assets.yaml")
    prices  = download_data(config)
    returns = compute_log_returns(prices)

    X = build_features(returns, config)
    y = build_target(returns, config)

    results = train_and_evaluate(X, y, config)
    print()
    print("Results:")
    print(results)

    wf_results = walk_forward_evaluation(
        X,
        y,
        config,
        train_fraction=0.6,
        step_size=20,
        test_window=20,
    )

    print()
    print(wf_results["metrics"])

    # Save results for later analysis on notebooks
    results['results'].to_csv("./results/predictions.csv")
    with open("results/metrics.json", "w") as f:
        json.dump(results['metrics'], f, indent=4)
        
    with open("./results/feature_names.json", "w") as f:
        json.dump(list(X.columns), f, indent=4)
        
    joblib.dump(results['models']['ridge'], "results/ridge_model.pkl")
    joblib.dump(results['models']['rf'], "results/rf_model.pkl")

    wf_results['results'].to_csv("./results/walk_forward_predictions.csv")
    with open("./results/walk_forward_metrics.json", "w") as f:
        json.dump(wf_results['metrics'], f, indent=4)
        
    print()
    print("INFO: predictions, metrics and models saved in ./results folder")
    print("INFO: run ./notebooks/02_visualization.ipynb for analysis")
    
    
if __name__ == "__main__":
    main()
