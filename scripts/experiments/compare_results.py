#!/usr/bin/env python
"""Compare results across all experiments."""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiment_results(results_dir):
    """Load all experiment results from the results directory."""
    results = {}
    results_path = Path(results_dir)
    
    for exp_dir in results_path.glob("*"):
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            results[exp_name] = {}
            
            # Load results for each task
            for task in ["tumor_type", "mgmt", "survival"]:
                result_file = exp_dir / f"{task}_results.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        results[exp_name][task] = json.load(f)
    
    return results

def create_comparison_table(results):
    """Create a comparison table of all experiments."""
    data = []
    
    for exp_name, exp_results in results.items():
        row = {"experiment": exp_name}
        
        for task, task_results in exp_results.items():
            if "accuracy" in task_results:
                row[f"{task}_accuracy"] = task_results["accuracy"]["mean"]
                row[f"{task}_std"] = task_results["accuracy"]["std"]
            elif "auc" in task_results:
                row[f"{task}_auc"] = task_results["auc"]["mean"]
                row[f"{task}_std"] = task_results["auc"]["std"]
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values("experiment")
    return df

def plot_experiment_comparisons(df):
    """Create plots comparing experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Dataset size comparison
    dataset_exps = df[df["experiment"].str.contains("dataset")].copy()
    if not dataset_exps.empty:
        ax = axes[0, 0]
        dataset_exps["size"] = dataset_exps["experiment"].str.extract(r"dataset_(\w+)_")[0]
        dataset_exps["lora"] = dataset_exps["experiment"].str.contains("lora")
        
        x = range(len(dataset_exps))
        width = 0.35
        
        for i, task in enumerate(["tumor_type", "mgmt", "survival"]):
            metric = f"{task}_accuracy" if task == "tumor_type" else f"{task}_auc"
            if metric in dataset_exps.columns:
                ax.bar([xi + i*width/3 for xi in x], dataset_exps[metric], 
                      width/3, label=task.replace("_", " ").title())
        
        ax.set_xlabel("Dataset Configuration")
        ax.set_ylabel("Performance")
        ax.set_title("Dataset Size Experiments")
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_exps["experiment"], rotation=45, ha="right")
        ax.legend()
    
    # Learning rate comparison
    lr_exps = df[df["experiment"].str.contains("lr_")].copy()
    if not lr_exps.empty:
        ax = axes[0, 1]
        lr_exps["lr"] = lr_exps["experiment"].str.extract(r"lr_(.+)")[0]
        
        for task in ["tumor_type", "mgmt", "survival"]:
            metric = f"{task}_accuracy" if task == "tumor_type" else f"{task}_auc"
            if metric in lr_exps.columns:
                ax.plot(lr_exps["lr"], lr_exps[metric], 'o-', label=task.replace("_", " ").title())
        
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Performance")
        ax.set_title("Learning Rate Experiments")
        ax.legend()
    
    # Task adapter comparison
    task_exps = df[df["experiment"].str.contains("task_")].copy()
    if not task_exps.empty:
        ax = axes[1, 0]
        task_exps["adapter"] = task_exps["experiment"].str.extract(r"task_(.+)")[0]
        
        x = range(len(task_exps))
        width = 0.35
        
        for i, task in enumerate(["tumor_type", "mgmt", "survival"]):
            metric = f"{task}_accuracy" if task == "tumor_type" else f"{task}_auc"
            if metric in task_exps.columns:
                ax.bar([xi + i*width/3 for xi in x], task_exps[metric], 
                      width/3, label=task.replace("_", " ").title())
        
        ax.set_xlabel("Adapter Type")
        ax.set_ylabel("Performance")
        ax.set_title("Task Adapter Experiments")
        ax.set_xticks(x)
        ax.set_xticklabels(task_exps["adapter"], rotation=45, ha="right")
        ax.legend()
    
    # Summary heatmap
    ax = axes[1, 1]
    metrics_df = df.set_index("experiment")
    metric_cols = [col for col in metrics_df.columns if "_accuracy" in col or "_auc" in col and "_std" not in col]
    
    if metric_cols:
        sns.heatmap(metrics_df[metric_cols], annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
        ax.set_title("Performance Heatmap")
    
    plt.tight_layout()
    plt.savefig("results/experiments/experiment_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig("results/experiments/experiment_comparison.pdf", bbox_inches="tight")

def main():
    """Main comparison function."""
    print("Loading experiment results...")
    results = load_experiment_results("results/experiments")
    
    print(f"Found {len(results)} experiments")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Save results
    print("\nSaving comparison table...")
    df.to_csv("results/experiments/experiment_comparison.csv", index=False)
    
    # Print summary
    print("\n=== EXPERIMENT COMPARISON ===")
    print(df.to_string(index=False))
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_experiment_comparisons(df)
    
    print("\nResults saved to:")
    print("  - results/experiments/experiment_comparison.csv")
    print("  - results/experiments/experiment_comparison.png")
    print("  - results/experiments/experiment_comparison.pdf")

if __name__ == "__main__":
    main()