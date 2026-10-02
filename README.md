# Cite Prob LLM

Research project for analyzing and comparing how different LLMs cite domains and URLs in responses generated from specific prompts.

The goal is to measure, at scale, how models such as OpenAI, Google AI, and Perplexity cite sources, convert those citations into links, and how that presence is distributed in terms of rank, frequency, and dominance within each scenario.

## Overview

This repository contains a complete pipeline for:

- loading raw prompt and AI response data;
- normalizing and cleaning the context of each prompt;
- calculating metrics by domain and by scenario;
- transforming this data into feature vectors;
- applying clustering to identify citation patterns;
- generating visualizations to compare models and prompts.

The approach combines citation auditing, feature engineering, and exploratory analysis using K-means, PCA, and t-SNE.

## Problem addressed

When an AI answers a question, the quality of the response is not determined only by the text itself. Other relevant factors include:

- which domains appear;
- how often they appear;
- whether they are converted into visible links;
- at what position they appear in the answer;
- whether the model is consistent across multiple iterations.

This project transforms those observations into a set of quantitative metrics to compare the behavior of different LLMs and prompts in the same context.

## Main pipeline

The project flow follows this sequence:

1. Collection of raw prompt and response data
2. Building the audit dataset by scenario
3. Feature engineering with V1 to V5 variables
4. Generation of vectors for clustering
5. K-means analysis and visualization

### Pipeline scripts

- `processing/02_build_vector_dataset.py`  
  Generates the audit table from the master dataset, with metrics such as URL presence, total citations, converted links, and average position.

- `processing/03_feature_engineering.py`  
  Creates the model features: presence rate, market share, position, conversion, and stability.

- `processing/04_kmeans.py`  
  Applies clustering by scenario to separate behavioral patterns across domains.

- `processing/05_plot.py`  
  Produces cluster plots and cluster profile summaries.

- `processing/07_visualizar_kmeans.py`  
  K-means visualization using PCA.

- `processing/08_visualizar_tsne.py`  
  Cluster visualization in t-SNE space.

- `processing/09_visualizar_tsne_3dimensoes.py`  
  3D visualization for deeper analysis.

## Repository structure

```text
.
├── data/
│   ├── master_dataset_tcc.csv
│   ├── gerador_vetores.csv
│   ├── processed/
│   ├── raw/
│   └── results/
├── processing/
│   ├── 02_build_vector_dataset.py
│   ├── 03_feature_engineering.py
│   ├── 04_kmeans.py
│   ├── 05_plot.py
│   ├── 05.1_plot.py
│   ├── 06_histogramas.py
│   ├── 07_visualizar_kmeans.py
│   ├── 08_visualizar_tsne.py
│   └── 09_visualizar_tsne_3dimensoes.py
├── notebook/
│   ├── choose_data.py
│   ├── extract_data_from_master.py
│   └── notebook.py
├── bright_data/
│   ├── api.py
│   ├── create_csv.py
│   └── merge.py
├── master/
├── notebook_data/
├── notebook_output/
├── .gitignore
├── README.md
└── ...
```

## Available data

In the folowing section we explain which data used but not it's structure itself since the data itself won't be shared to the public but you can contact me if you have any questions regarding how I collected, structured, and processed it. After that, the workflow can begin directly from the files under `data/` and `master/`.

Main expected files:

- `data/master_dataset_tcc.csv`  
  Master dataset with prompt results and citation metrics.

- `data/gerador_vetores.csv`  
  Audit table organized by domain and scenario.

- `data/processed/`  
  Final vectors by scenario, ready for clustering.

- `data/results/`  
  Analytical outputs and visualizations.

## Main metrics

The features built by the project represent a signature of presence and relevance for each URL within a given scenario:

- V1: presence rate
- V2: market share / dominance
- V3A: important average position
- V3B: normalized position ignoring noise
- V4: conversion rate to link
- V5: stability of positional deviation

These variables allow domains to be compared more robustly than by counting raw citations alone.

## How to run

### 1) Environment setup

Recommended: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn plotly
```

### 2) Run the pipeline

```bash
python processing/02_build_vector_dataset.py
python processing/03_feature_engineering.py
python processing/04_kmeans.py
```

After that, you can run the visualization scripts:

```bash
python processing/05_plot.py
python processing/07_visualizar_kmeans.py
python processing/08_visualizar_tsne.py
python processing/09_visualizar_tsne_3dimensoes.py
```

## Expected outputs

Outputs include:

- CSV files with URL and scenario audits;
- final vectors in `data/processed/`;
- clustering charts;
- cluster profiles by LLM and prompt;
- visual comparisons among different information sources.

## Use cases

This project is useful for:

- comparing citation quality across AI models;
- evaluating which domains tend to receive more attention;
- identifying source legitimization patterns;
- studying citation consistency across multiple responses;
- producing digital reputation and visibility analysis in AI search contexts.

## Notes

- The project is structured to handle one scenario at a time, separating LLM + prompt to avoid mixing different markets and patterns.
- The feature engineering script distinguishes real zero cases from artificial zero cases caused by insufficient data.
- The analysis is especially useful when moving beyond simple citation counts and trying to understand the probability of source presence, conversion, and stability.

## Status

Project yielded good results, with a functional pipeline for processing, feature engineering, clustering, and visualization, however it can still have upgrades which will be our next step. Last update was (1/10/2026 as dd/mm/yy)

---
