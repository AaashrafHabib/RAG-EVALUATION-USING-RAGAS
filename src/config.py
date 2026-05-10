from dataclasses import dataclass, field
from typing import Literal

AWS_REGION = "eu-west-1"


@dataclass
class ChunkingConfig:
    strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class EmbeddingConfig:
    model_name: str = "amazon.titan-embed-text-v2:0"
    provider: Literal["bedrock", "huggingface"] = "bedrock"


@dataclass
class RetrieverConfig:
    method: Literal["naive", "hybrid"] = "naive"
    top_k: int = 5
    bm25_weight: float = 0.3


@dataclass
class LLMConfig:
    model_name: str = "eu.anthropic.claude-sonnet-4-20250514-v1:0"
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class PipelineConfig:
    name: str = "default"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def __str__(self):
        return (
            f"{self.name} | chunk={self.chunking.strategy}({self.chunking.chunk_size}) "
            f"| embed={self.embedding.model_name} "
            f"| retrieval={self.retriever.method}(k={self.retriever.top_k}) "
            f"| llm={self.llm.model_name}"
        )


EXPERIMENT_CONFIGS = [
    # --- Chunking variations (with Claude Sonnet 4) ---
    PipelineConfig(
        name="baseline_small_chunks",
        chunking=ChunkingConfig(strategy="recursive", chunk_size=256, chunk_overlap=30),
        embedding=EmbeddingConfig(model_name="amazon.titan-embed-text-v2:0", provider="bedrock"),
        retriever=RetrieverConfig(method="naive", top_k=5),
        llm=LLMConfig(model_name="eu.anthropic.claude-sonnet-4-20250514-v1:0"),
    ),
    PipelineConfig(
        name="baseline_medium_chunks",
        chunking=ChunkingConfig(strategy="recursive", chunk_size=512, chunk_overlap=50),
        embedding=EmbeddingConfig(model_name="amazon.titan-embed-text-v2:0", provider="bedrock"),
        retriever=RetrieverConfig(method="naive", top_k=5),
        llm=LLMConfig(model_name="eu.anthropic.claude-sonnet-4-20250514-v1:0"),
    ),
    PipelineConfig(
        name="baseline_large_chunks",
        chunking=ChunkingConfig(strategy="recursive", chunk_size=1024, chunk_overlap=100),
        embedding=EmbeddingConfig(model_name="amazon.titan-embed-text-v2:0", provider="bedrock"),
        retriever=RetrieverConfig(method="naive", top_k=5),
        llm=LLMConfig(model_name="eu.anthropic.claude-sonnet-4-20250514-v1:0"),
    ),
    PipelineConfig(
        name="semantic_chunking",
        chunking=ChunkingConfig(strategy="semantic", chunk_size=512, chunk_overlap=0),
        embedding=EmbeddingConfig(model_name="amazon.titan-embed-text-v2:0", provider="bedrock"),
        retriever=RetrieverConfig(method="naive", top_k=5),
        llm=LLMConfig(model_name="eu.anthropic.claude-sonnet-4-20250514-v1:0"),
    ),
    # --- Retrieval variations ---
    PipelineConfig(
        name="hybrid_search",
        chunking=ChunkingConfig(strategy="recursive", chunk_size=512, chunk_overlap=50),
        embedding=EmbeddingConfig(model_name="amazon.titan-embed-text-v2:0", provider="bedrock"),
        retriever=RetrieverConfig(method="hybrid", top_k=5, bm25_weight=0.3),
        llm=LLMConfig(model_name="eu.anthropic.claude-sonnet-4-20250514-v1:0"),
    ),
    PipelineConfig(
        name="hybrid_heavy_bm25",
        chunking=ChunkingConfig(strategy="recursive", chunk_size=512, chunk_overlap=50),
        embedding=EmbeddingConfig(model_name="amazon.titan-embed-text-v2:0", provider="bedrock"),
        retriever=RetrieverConfig(method="hybrid", top_k=5, bm25_weight=0.6),
        llm=LLMConfig(model_name="eu.anthropic.claude-sonnet-4-20250514-v1:0"),
    ),
    # --- Embedding variations ---
    PipelineConfig(
        name="huggingface_embed",
        chunking=ChunkingConfig(strategy="recursive", chunk_size=512, chunk_overlap=50),
        embedding=EmbeddingConfig(model_name="all-MiniLM-L6-v2", provider="huggingface"),
        retriever=RetrieverConfig(method="naive", top_k=5),
        llm=LLMConfig(model_name="eu.anthropic.claude-sonnet-4-20250514-v1:0"),
    ),
    # --- LLM variations ---
    PipelineConfig(
        name="claude_haiku_3",
        chunking=ChunkingConfig(strategy="recursive", chunk_size=512, chunk_overlap=50),
        embedding=EmbeddingConfig(model_name="amazon.titan-embed-text-v2:0", provider="bedrock"),
        retriever=RetrieverConfig(method="naive", top_k=5),
        llm=LLMConfig(model_name="eu.anthropic.claude-3-haiku-20240307-v1:0"),
    ),
    PipelineConfig(
        name="claude_haiku_4_5",
        chunking=ChunkingConfig(strategy="recursive", chunk_size=512, chunk_overlap=50),
        embedding=EmbeddingConfig(model_name="amazon.titan-embed-text-v2:0", provider="bedrock"),
        retriever=RetrieverConfig(method="naive", top_k=5),
        llm=LLMConfig(model_name="eu.anthropic.claude-haiku-4-5-20251001-v1:0"),
    ),
    # --- Full optimized ---
    PipelineConfig(
        name="full_optimized",
        chunking=ChunkingConfig(strategy="semantic", chunk_size=512, chunk_overlap=0),
        embedding=EmbeddingConfig(model_name="amazon.titan-embed-text-v2:0", provider="bedrock"),
        retriever=RetrieverConfig(method="hybrid", top_k=7, bm25_weight=0.3),
        llm=LLMConfig(model_name="eu.anthropic.claude-sonnet-4-20250514-v1:0"),
    ),
]
