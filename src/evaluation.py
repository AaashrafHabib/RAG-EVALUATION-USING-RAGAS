import time
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_aws import ChatBedrock, BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.config import PipelineConfig, AWS_REGION
from src.rag_pipeline import RAGPipeline


def get_ragas_llm(model_id: str = "eu.anthropic.claude-sonnet-4-20250514-v1:0"):
    llm = ChatBedrock(
        model_id=model_id,
        region_name=AWS_REGION,
        model_kwargs={"temperature": 0.0, "max_tokens": 2048},
    )
    return LangchainLLMWrapper(llm)


def get_ragas_embeddings(model_id: str = "amazon.titan-embed-text-v2:0"):
    embeddings = BedrockEmbeddings(
        model_id=model_id,
        region_name=AWS_REGION,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def _extract_score(value) -> float:
    """Extract a single float from RAGAS result (handles both list and scalar)."""
    if isinstance(value, list):
        return float(np.mean(value))
    return float(value)


def run_evaluation(
    pipeline: RAGPipeline,
    test_questions: List[str],
    ground_truths: List[str],
    judge_model: str = "eu.anthropic.claude-sonnet-4-20250514-v1:0",
) -> Dict:
    """Run the RAG pipeline on test questions and evaluate with RAGAS."""

    print(f"\n{'='*60}")
    print(f"Evaluating: {pipeline.config.name}")
    print(f"{'='*60}")

    results = []
    total_latency = 0

    for i, (question, gt) in enumerate(zip(test_questions, ground_truths)):
        print(f"  Processing question {i+1}/{len(test_questions)}...", end="\r")
        result = pipeline.query(question)
        result["ground_truth"] = gt
        results.append(result)
        total_latency += result["latency"]

    avg_latency = total_latency / len(test_questions)
    print(f"  All {len(test_questions)} questions processed. Avg latency: {avg_latency:.2f}s")

    ragas_dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    print(f"  Running RAGAS evaluation (judge: {judge_model})...")
    ragas_llm = get_ragas_llm(judge_model)
    ragas_embeddings = get_ragas_embeddings()

    start = time.time()
    ragas_result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    eval_time = time.time() - start
    print(f"  RAGAS evaluation completed in {eval_time:.1f}s")

    scores = {
        "config_name": pipeline.config.name,
        "faithfulness": _extract_score(ragas_result["faithfulness"]),
        "answer_relevancy": _extract_score(ragas_result["answer_relevancy"]),
        "context_precision": _extract_score(ragas_result["context_precision"]),
        "context_recall": _extract_score(ragas_result["context_recall"]),
        "avg_latency": avg_latency,
        "total_time": total_latency,
        "eval_time": eval_time,
        "num_questions": len(test_questions),
        "judge_model": judge_model,
        "config_details": str(pipeline.config),
    }

    print(f"\n  Results for {pipeline.config.name}:")
    print(f"    Faithfulness:       {scores['faithfulness']:.4f}")
    print(f"    Answer Relevancy:   {scores['answer_relevancy']:.4f}")
    print(f"    Context Precision:  {scores['context_precision']:.4f}")
    print(f"    Context Recall:     {scores['context_recall']:.4f}")
    print(f"    Avg Latency:        {scores['avg_latency']:.2f}s")

    return scores


def run_all_experiments(
    configs: List[PipelineConfig],
    documents,
    test_questions: List[str],
    ground_truths: List[str],
    output_dir: str = "data/results",
    judge_model: str = "eu.anthropic.claude-sonnet-4-20250514-v1:0",
) -> pd.DataFrame:
    """Run all experiment configurations and save results."""

    all_scores = []

    for config in configs:
        try:
            pipeline = RAGPipeline(config)
            pipeline.load_documents(documents)
            pipeline.chunk_documents()
            pipeline.build_index()

            scores = run_evaluation(pipeline, test_questions, ground_truths, judge_model)
            all_scores.append(scores)

            pipeline.cleanup()
        except Exception as e:
            print(f"\n  ERROR with config '{config.name}': {e}")
            all_scores.append({
                "config_name": config.name,
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
                "avg_latency": None,
                "error": str(e),
            })

    df = pd.DataFrame(all_scores)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"benchmark_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    json_path = output_path / f"benchmark_results_{timestamp}.json"
    df.to_json(json_path, orient="records", indent=2)

    return df


def compare_judges(
    pipeline: RAGPipeline,
    test_questions: List[str],
    ground_truths: List[str],
    judge_models: List[str],
) -> pd.DataFrame:
    """Compare RAGAS scores across different LLM judges."""

    pipeline_results = []
    for question, gt in zip(test_questions, ground_truths):
        result = pipeline.query(question)
        result["ground_truth"] = gt
        pipeline_results.append(result)

    ragas_dataset = Dataset.from_dict({
        "question": [r["question"] for r in pipeline_results],
        "answer": [r["answer"] for r in pipeline_results],
        "contexts": [r["contexts"] for r in pipeline_results],
        "ground_truth": [r["ground_truth"] for r in pipeline_results],
    })

    results_all = []
    for judge_model in judge_models:
        print(f"\n  Evaluating with judge: {judge_model}")
        ragas_llm = get_ragas_llm(judge_model)
        ragas_embeddings = get_ragas_embeddings()

        ragas_result = evaluate(
            ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        results_all.append({
            "judge_model": judge_model,
            "faithfulness": _extract_score(ragas_result["faithfulness"]),
            "answer_relevancy": _extract_score(ragas_result["answer_relevancy"]),
            "context_precision": _extract_score(ragas_result["context_precision"]),
            "context_recall": _extract_score(ragas_result["context_recall"]),
        })

    return pd.DataFrame(results_all)
