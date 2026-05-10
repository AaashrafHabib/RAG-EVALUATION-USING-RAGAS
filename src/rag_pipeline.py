import time
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import PipelineConfig, AWS_REGION


class RAGPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.vectorstore = None
        self.documents = None
        self.chunks = None
        self.bm25 = None

    def load_documents(self, documents: List[Document]):
        self.documents = documents
        return self

    def chunk_documents(self) -> List[Document]:
        if self.config.chunking.strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        elif self.config.chunking.strategy == "fixed":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap,
                separators=[""],
            )
        elif self.config.chunking.strategy == "semantic":
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                embeddings = self._get_embeddings()
                splitter = SemanticChunker(embeddings)
            except ImportError:
                print("langchain_experimental not installed, falling back to recursive")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunking.chunk_size,
                    chunk_overlap=self.config.chunking.chunk_overlap,
                )
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.chunking.strategy}")

        self.chunks = splitter.split_documents(self.documents)
        print(f"  [{self.config.name}] Created {len(self.chunks)} chunks")
        return self.chunks

    def _get_embeddings(self):
        if self.config.embedding.provider == "bedrock":
            from langchain_aws import BedrockEmbeddings
            return BedrockEmbeddings(
                model_id=self.config.embedding.model_name,
                region_name=AWS_REGION,
            )
        elif self.config.embedding.provider == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=self.config.embedding.model_name)
        else:
            raise ValueError(f"Unknown embedding provider: {self.config.embedding.provider}")

    def _get_llm(self):
        from langchain_aws import ChatBedrock
        return ChatBedrock(
            model_id=self.config.llm.model_name,
            region_name=AWS_REGION,
            model_kwargs={
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
            },
        )

    def build_index(self):
        embeddings = self._get_embeddings()
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            collection_name=self.config.name.replace(" ", "_"),
        )

        if self.config.retriever.method in ("hybrid",):
            tokenized_chunks = [doc.page_content.split() for doc in self.chunks]
            self.bm25 = BM25Okapi(tokenized_chunks)

        print(f"  [{self.config.name}] Index built with {len(self.chunks)} vectors")
        return self

    def retrieve(self, query: str) -> List[Document]:
        k = self.config.retriever.top_k

        if self.config.retriever.method == "naive":
            return self.vectorstore.similarity_search(query, k=k)

        elif self.config.retriever.method == "hybrid":
            dense_results = self.vectorstore.similarity_search(query, k=k * 2)
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_indices = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[:k * 2]
            bm25_results = [self.chunks[i] for i in bm25_top_indices]

            seen = set()
            combined = []
            for doc in dense_results:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    combined.append(doc)
            for doc in bm25_results:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    combined.append(doc)
            return combined[:k]

        return []

    def generate(self, query: str, context_docs: List[Document]) -> Tuple[str, float]:
        llm = self._get_llm()

        context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
        prompt = f"""Based on the following context, answer the question accurately and concisely.
If the context doesn't contain enough information to answer, say so explicitly.

Context:
{context}

Question: {query}

Answer:"""

        start = time.time()
        response = llm.invoke(prompt)
        latency = time.time() - start

        return response.content, latency

    def query(self, question: str) -> dict:
        context_docs = self.retrieve(question)
        answer, latency = self.generate(question, context_docs)
        return {
            "question": question,
            "answer": answer,
            "contexts": [doc.page_content for doc in context_docs],
            "latency": latency,
        }

    def cleanup(self):
        if self.vectorstore:
            try:
                self.vectorstore.delete_collection()
            except Exception:
                pass
