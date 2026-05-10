# RAGAS Benchmark : Évaluation avancée de pipelines RAG

Projet d'évaluation systématique de pipelines RAG (Retrieval-Augmented Generation) utilisant le framework RAGAS, appliqué au corpus de l'EU AI Act.

## Structure

```
├── data/
│   ├── corpus/          # Documents sources (EU AI Act)
│   ├── testset/         # Questions + ground truth annotées manuellement
│   └── results/         # Résultats des benchmarks (CSV, PNG)
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_rag_pipeline.ipynb
│   ├── 03_ragas_deep_dive.ipynb
│   └── 04_analysis_visualization.ipynb
├── src/
│   ├── config.py        # Configurations des 10 expériences
│   ├── rag_pipeline.py  # Pipeline RAG modulaire
│   ├── evaluation.py    # Orchestration RAGAS
│   ├── data_loader.py   # Chargement corpus/testset
│   └── visualization.py # Graphiques (radar, heatmap, etc.)
├── run_benchmark.py     # Script principal
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
# AWS credentials doivent être configurées (~/.aws/credentials)
aws sts get-caller-identity  # Vérifier l'accès
```

## Utilisation

```bash
# Test rapide (4 configs)
python run_benchmark.py --configs quick

# Benchmark complet (10 configs)
python run_benchmark.py --configs all

# Changer le modèle juge
python run_benchmark.py --configs quick --judge anthropic.claude-haiku-4-5-20251001-v1:0
```

## Infrastructure

**AWS Bedrock (eu-west-1)** — Pas de clé API, authentification IAM.

## Expériences

10 configurations testées, variant :
- **Chunking** : fixed (256/512/1024), recursive, semantic
- **Embeddings** : Amazon Titan Embed Text V2 (Bedrock), HuggingFace all-MiniLM-L6-v2 (local)
- **Retrieval** : naive (dense), hybrid (BM25 + dense)
- **LLM générateur** : Claude Sonnet 4, Claude Haiku 4.5, Claude Haiku 3
- **LLM juge RAGAS** : Claude Sonnet 4 (par défaut)

## Métriques RAGAS

- **Faithfulness** : la réponse est-elle fidèle au contexte récupéré ?
- **Answer Relevancy** : la réponse est-elle pertinente par rapport à la question ?
- **Context Precision** : les chunks pertinents sont-ils bien classés ?
- **Context Recall** : tous les éléments nécessaires ont-ils été récupérés ?

## Corpus

EU AI Act — choisi pour :
- Récence (2024) → faible contamination LLM
- Structure claire (articles, annexes) → vérification factuelle facilitée
- Pertinence aivancity (Tech × Business × Éthique)
