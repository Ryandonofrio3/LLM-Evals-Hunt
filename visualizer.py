import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict
import os

class ResultVisualizer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir

    def create_performance_graphs(self, stats: Dict):
        # Create figures directory if it doesn't exist
        os.makedirs(f"{self.results_dir}/figures", exist_ok=True)

        # Prepare data for plotting
        model_data = []
        for model, perf in stats["model_performance"].items():
            model_data.append({
                "model": model,
                "accuracy": perf["accuracy"],
                "correct_answers": perf["correct_answers"],
                "errors": perf["errors"]
            })
        
        df = pd.DataFrame(model_data)

        # Create accuracy bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="model", y="accuracy")
        plt.title("Model Accuracy Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/accuracy_comparison.png")
        plt.close()

        # Create error comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="model", y="errors", color="red")
        plt.title("Model Errors Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/error_comparison.png")
        plt.close()

        return {
            "accuracy_plot": f"{self.results_dir}/figures/accuracy_comparison.png",
            "error_plot": f"{self.results_dir}/figures/error_comparison.png"
        } 