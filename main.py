from puzzles import Model
from puzzle_data import PUZZLE_DATABASE
from evaluator import PuzzleEvaluator

def main():
    models_to_test = [
        Model.GPT4o_MINI,
        Model.GPT4o,
        Model.GPT4_Turbo,
        # Model.CLAUDE3,
    ]

    evaluator = PuzzleEvaluator(
        puzzles=PUZZLE_DATABASE,
        models=models_to_test
    )

    results = evaluator.run_evaluation()

    print("\nEvaluation Results:")
    print("-" * 50)
    for model, stats in results["model_performance"].items():
        print(f"\nModel: {model}")
        print(f"Accuracy: {stats['accuracy']:.2%}")
        print(f"Correct: {stats['correct_answers']}/{stats['total_attempts']}")
        print(f"Errors: {stats['errors']}")

    print(f"\nVisualizations saved to:")
    for plot_type, path in results["visualization_paths"].items():
        print(f"{plot_type}: {path}")

if __name__ == "__main__":
    main() 