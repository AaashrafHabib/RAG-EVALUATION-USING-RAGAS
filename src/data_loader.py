import json
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document


def load_corpus_from_texts(corpus_dir: str = "data/corpus") -> List[Document]:
    """Load all .txt files from the corpus directory as LangChain Documents."""
    docs = []
    corpus_path = Path(corpus_dir)

    for file_path in sorted(corpus_path.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")
        docs.append(Document(
            page_content=content,
            metadata={"source": file_path.name},
        ))

    print(f"Loaded {len(docs)} documents from {corpus_dir}")
    return docs


def load_testset(testset_path: str = "data/testset/testset.json") -> Tuple[List[str], List[str]]:
    """Load test questions and ground truths from JSON file."""
    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    questions = [item["question"] for item in testset]
    ground_truths = [item["ground_truth"] for item in testset]

    print(f"Loaded {len(questions)} test questions from {testset_path}")
    return questions, ground_truths


def load_human_scores(path: str = "data/testset/human_scores.json") -> List[dict]:
    """Load human evaluation scores."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
