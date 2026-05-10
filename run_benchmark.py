"""
Run the full RAGAS benchmark pipeline using AWS Bedrock.
Usage: python run_benchmark.py [--configs all|quick]
"""
import argparse

from src.config import EXPERIMENT_CONFIGS
from src.evaluation import run_all_experiments
from src.data_loader import load_corpus_from_texts, load_testset
from src.visualization import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS benchmark (AWS Bedrock)")
    parser.add_argument("--configs", choices=["all", "quick"], default="quick",
                       help="Run all configs or just a quick subset (first 4)")
    parser.add_argument("--judge", default="eu.anthropic.claude-sonnet-4-20250514-v1:0",
                       help="Bedrock inference profile ID for RAGAS judge")
    args = parser.parse_args()

    print("Loading data...")
    documents = load_corpus_from_texts("data/corpus")
    questions, ground_truths = load_testset("data/testset/testset.json")

    if args.configs == "all":
        configs = EXPERIMENT_CONFIGS
    else:
        configs = EXPERIMENT_CONFIGS[:4]

    print(f"\nRunning {len(configs)} configurations on {len(questions)} questions...")
    print(f"Judge model: {args.judge}")
    print(f"Region: eu-west-1\n")

    results_df = run_all_experiments(
        configs=configs,
        documents=documents,
        test_questions=questions,
        ground_truths=ground_truths,
        output_dir="data/results",
        judge_model=args.judge,
    )

    print("\nGenerating visualizations...")
    from pathlib import Path
    csv_files = sorted(Path("data/results").glob("benchmark_results_*.csv"))
    if csv_files:
        generate_all_plots(str(csv_files[-1]))

    print("\nDone! Check data/results/ for outputs.")


if __name__ == "__main__":
    main()
