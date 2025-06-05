# Text Analytics for Emotion Classification and Biomedical Named Entity Recognition

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/uchit-bhadauriya-a96478204)

## Overview

This repository presents a comprehensive exploration of text analytics techniques for **emotion classification in tweets** and **biomedical named entity recognition (NER)**. It features classic machine learning models (Naive Bayes), deep learning (CNN, RoBERTa), and topic modeling (NMF), accompanied by in-depth evaluation and analysis.

**Core Tasks:**
- **Task 1:** Emotion classification in tweets using Naive Bayes and CNN
- **Task 2:** Biomedical NER with RoBERTa and entity similarity analysis using BERT/Word2Vec

---

## Table of Contents

- [Project Structure](#project-structure)
- [Task 1: Emotion Classification](#task-1-emotion-classification)
  - [1.1 Naive Bayes Classifier](#11-naive-bayes-classifier)
  - [1.2 CNN Classifier](#12-cnn-classifier)
  - [1.3 Evaluation & Discussion](#13-evaluation--discussion)
  - [1.4 Topic Modeling with NMF](#14-topic-modeling-with-nmf)
- [Task 2: Biomedical NER](#task-2-biomedical-ner)
  - [2.1 RoBERTa-based Sequence Tagger](#21-roberta-based-sequence-tagger)
  - [2.2 Evaluation & Results](#22-evaluation--results)
  - [2.3 Disease Entity Similarity Analysis](#23-disease-entity-similarity-analysis)
- [Installation & Usage](#installation--usage)
- [Results Summary](#results-summary)
- [References](#references)
- [Contact](#contact)

---

## Task 1: Emotion Classification

### 1.1 Naive Bayes Classifier

- **Algorithm:** Multinomial Naive Bayes  
- **Preprocessing:** Tokenization, lowercasing, removal of URLs/mentions/punctuation/numbers, stopword removal, lemmatization  
- **Feature Representation:** Count Vectorizer  
- **Strengths:** Fast, scalable, robust to irrelevant features  
- **Limitations:** Assumes feature independence, limited with correlated features or imbalanced data

### 1.2 CNN Classifier

- **Architecture:** Embedding layer (with GloVe), convolutional layers, pooling, dense output  
- **Advantages:** Learns local and hierarchical features, leverages pre-trained word embeddings  
- **Preprocessing:** Similar to Naive Bayes, plus sequence padding and embedding preparation

### 1.3 Evaluation & Discussion

**Metrics Used:**
- Accuracy, Precision, Recall, F1-Score

| Model        | Accuracy | F1-Score |
|--------------|----------|----------|
| Naive Bayes  | ~67%     | 0.65     |
| CNN          | ~70%     | 0.63     |

- **CNN outperforms Naive Bayes** in capturing text nuance.
- Both models suffer from class imbalance; further improvements possible with data augmentation, class weighting, and advanced feature engineering.

### 1.4 Topic Modeling with NMF

- **Method:** Non-negative Matrix Factorization (NMF) on TF-IDF vectors
- **Purpose:** Discover themes/topics within emotions (joy, optimism, etc.)
- **Result:**  
  - **Joy:** Themes include celebrations, humor, social events.
  - **Optimism:** Themes of overcoming challenges, inspirational quotes.

---

## Task 2: Biomedical Named Entity Recognition

### 2.1 RoBERTa-based Sequence Tagger

- **Approach:** Fine-tuned RoBERTa model using BIO tagging for NER  
- **Features:** Contextual embeddings, POS tags, character-level patterns

### 2.2 Evaluation & Results

**Validation Set Results:**

| Entity Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| 0           | 0.99      | 0.98   | 0.98     |
| 1           | 0.84      | 0.91   | 0.88     |
| ...         | ...       | ...    | ...      |

- **High precision and recall** for key entity types  
- Common errors: ambiguous boundaries, entity type confusion, rare entity misses

### 2.3 Disease Entity Similarity Analysis

- **Techniques:** BERT and Word2Vec embeddings, cosine similarity  
- **Goal:** Find diseases most/least similar to a given query (e.g., “dyskinesia”)  
- **Insight:** BERT provides richer context, Word2Vec is faster but less nuanced

---

## Installation & Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
## Install dependencies:

- Python ≥ 3.7
- Jupyter Notebook
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, nltk, gensim, torch, transformers, etc.
- (Install using `pip install -r requirements.txt` if provided.)

## Run Notebooks:

Open notebooks with Jupyter and run sequentially for each task.

---

## Results Summary

**Emotion Classification:**  
CNN outperformed Naive Bayes with higher accuracy and F1-score, thanks to feature learning and transfer learning (GloVe).

**Topic Modeling:**  
NMF effectively revealed interpretable themes for emotions like joy and optimism.

**Biomedical NER:**  
RoBERTa achieved state-of-the-art performance, with strong precision and recall on biomedical entities.

**Entity Similarity:**  
BERT embeddings gave better semantic matching than Word2Vec for disease similarity analysis.

---

## References

- IBM, [Naive Bayes](https://www.ibm.com/topics/naive-bayes)
- Wikipedia, [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- Medium, [NMF Explained](https://medium.com/@neri.vvo/non-negative-matrix-factorization-explained-practical-how-to-guide-in-python-c6372f2f6779)
- Medium, [RoBERTa: A Robustly Optimized BERT Approach](https://towardsdatascience.com/roberta-1ef07226c8d8)

---

## Contact

Created by [Uchit Bhadauriya](https://www.linkedin.com/in/uchit-bhadauriya-a96478204)  
Feel free to connect for questions, collaborations, or feedback!

---

✨ *Feel free to fork, star, or open an issue for suggestions and improvements!*
