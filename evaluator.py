from typing import List, Dict
from puzzles import Puzzle, Model, PuzzleSolver
import pandas as pd
from datetime import datetime
import json
import os
from visualizer import ResultVisualizer

class PuzzleEvaluator:
    def __init__(self, puzzles: List[Puzzle], models: List[Model]):
        self.puzzles = puzzles
        self.models = models
        self.solver = PuzzleSolver()
        self.results = []
        self.visualizer = ResultVisualizer()

    def run_evaluation(self) -> Dict:
        for puzzle in self.puzzles:
            for model in self.models:
                result = self.solver.solve_puzzle(puzzle, model)
                print(f"Response for {model.value.name}: {result['raw_response']}")
                self.results.append(result)
        
        return self._generate_report()

    def _generate_report(self) -> Dict:
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        stats = {
            "total_puzzles": len(self.puzzles),
            "total_attempts": len(self.results),
            "timestamp": datetime.now().isoformat(),
            "model_performance": {}
        }

        for model in self.models:
            model_results = df[df['model'] == model.value.name]
            stats["model_performance"][model.value.name] = {
                "correct_answers": int(model_results['is_correct'].sum()),
                "total_attempts": len(model_results),
                "accuracy": float(model_results['is_correct'].mean()),
                "errors": len(model_results[model_results['error'].notna()])
            }

        # Generate visualizations
        visualization_paths = self.visualizer.create_performance_graphs(stats)
        stats["visualization_paths"] = visualization_paths
        
        # Save results
        self._save_results(stats)
        
        return stats

    def _save_results(self, stats: Dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        with open(f"results/detailed_results_{timestamp}.json", "w") as f:
            json.dump(self.results, f, indent=2)
            
        # Save statistics
        with open(f"results/stats_{timestamp}.json", "w") as f:
            json.dump(stats, f, indent=2) 