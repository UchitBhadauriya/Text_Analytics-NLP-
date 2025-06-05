# Text Analytics & Emotion Classification

A comprehensive project applying classical machine learning, deep learning (CNN), and advanced NLP (RoBERTa) to emotion classification, topic modeling, and named entity recognition on social text data.

[Connect on LinkedIn](https://www.linkedin.com/in/uchit-bhadauriya-a96478204)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Tasks & Methods](#tasks--methods)
- [Getting Started](#getting-started)
- [Results & Visualizations](#results--visualizations)
- [Improvements & Future Work](#improvements--future-work)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project explores an end-to-end NLP workflow including:
- **Emotion classification** with Naive Bayes and CNNs
- **Topic modeling** using Non-negative Matrix Factorization (NMF)
- **Named Entity Recognition (NER)** with a transformer (RoBERTa)
- **Biomedical entity similarity analysis**

All results are demonstrated on imbalanced, real-world text datasets, with interpretability and practical evaluation in focus.

---

## Project Structure


---

## Tasks & Methods

### 1. Emotion Classification

#### Naive Bayes Classifier
- **Preprocessing:** Tokenization, lowercasing, noise removal (URLs, mentions, hashtags, punctuation, numbers), stopword removal, lemmatization.
- **Features:** CountVectorizer (word counts).
- **Strengths:** Fast, simple, good for small/imbalanced datasets.
- **Limitations:** Independence assumption, struggles with feature correlations.

#### Convolutional Neural Network (CNN)
- **Preprocessing:** Same as above + sequence padding for input.
- **Features:** Pre-trained embeddings (e.g., GloVe).
- **Strengths:** Learns local/hierarchical features, benefits from transfer learning.
- **Limitations:** Needs more data, higher computation.

---

### 2. Topic Modeling (NMF)

- **Steps:** Tokenization, lowercasing, stopword removal, TF-IDF transformation.
- **Goal:** Extract latent topics/word groups for each emotion label, providing insight into how emotions are expressed.

---

### 3. Named Entity Recognition (NER) with RoBERTa

- **Architecture:** Fine-tuned RoBERTa transformer with BIO tagging.
- **Features:** RoBERTa embeddings, POS tags, character/context features.
- **Strengths:** High contextual understanding, state-of-the-art results.

---

### 4. Biomedical Entity Similarity

- **Techniques:** BERT embeddings, Word2Vec, cosine similarity.
- **Goal:** Identify most/least similar disease entities in biomedical text.

---

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/<repo-name>.git
    cd <repo-name>
    ```

2. **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run notebooks:**
    - Launch Jupyter and open any notebook of interest.
    - Execute cells to preprocess, train, and evaluate models.

---

## Results & Visualizations

### 1. Naive Bayes Classifier: Test Performance

<img src="images/Screenshot_2025-06-05_at_11.14.45_AM.png" width="600">

**Explanation:**  
This classification report shows Naive Bayes performance for each emotion.  
- Best on class 0 (F1: 0.75); weaker on minority classes due to class imbalance.  
- **Overall accuracy:** 67%

---

### 2. CNN Model: Training and Validation Curves

<img src="images/Screenshot_2025-06-05_at_11.14.57_AM.png" width="600">

**Explanation:**  
- Train/validation loss decreases steadily, showing effective model learning.
- **Final test accuracy:** 70.23%
- **F1 Score:** 0.6295

---

### 3. Emotion Imbalance in Training Data

<img src="images/Screenshot_2025-06-05_at_11.15.10_AM.png" width="600">

**Explanation:**  
- Class frequencies in the training set, showing a strong imbalance (anger > sadness > joy > optimism).

---

### 4. Topic Modeling (NMF): 'Joy' Topics & Wordclouds

<img src="images/Screenshot_2025-06-05_at_11.15.16_AM.png" width="600">

- **Extracted Topics:**  
  Top words for 5 distinct topics within 'Joy', e.g. "lively", "broadcast", "musically".

<img src="images/Screenshot_2025-06-05_at_11.16.47_AM.png" width="600">
<img src="images/Screenshot_2025-06-05_at_11.16.53_AM.png" width="600">
<img src="images/Screenshot_2025-06-05_at_11.17.00_AM.png" width="600">
<img src="images/Screenshot_2025-06-05_at_11.17.06_AM.png" width="600">
<img src="images/Screenshot_2025-06-05_at_11.17.12_AM.png" width="600">

**Explanation:**  
Each wordcloud visualizes the most significant words for a given 'Joy' topic, helping interpret how positive emotions are expressed in social data.

---

### 5. Topic Modeling (NMF): 'Optimism' Topics & Wordclouds

<img src="images/Screenshot_2025-06-05_at_11.17.31_AM.png" width="600">
<img src="images/Screenshot_2025-06-05_at_11.17.34_AM.png" width="600">
<img src="images/Screenshot_2025-06-05_at_11.17.37_AM.png" width="600">
<img src="images/Screenshot_2025-06-05_at_11.17.40_AM.png" width="600">
<img src="images/Screenshot_2025-06-05_at_11.17.42_AM.png" width="600">

**Explanation:**  
Wordclouds for each optimism-related topic, e.g. "worry", "leadership", "fear", "life", "start", "advice", "you're", "good", "make".  
They highlight the diverse ways optimism is discussed in the dataset.

---

### 6. Named Entity Recognition (RoBERTa): Model Metrics

<img src="images/Screenshot_2025-06-05_at_11.17.59_AM.png" width="600">

**Explanation:**  
- RoBERTa NER model achieves high precision and recall for entity extraction in biomedical text.
- Macro F1: 0.81, Weighted F1: 0.97

---

## Improvements & Future Work

- Add data augmentation for better handling of class imbalance.
- Explore more advanced neural models (e.g., LSTM, transformer-based emotion classification).
- Fine-tune hyperparameters (using grid/random search).
- Extend NER to more entity types or domains.
- Add full end-to-end deployment demo (e.g., API or web interface).

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Contact

**Uchit Bhadauriya**  
[LinkedIn Profile](https://www.linkedin.com/in/uchit-bhadauriya-a96478204)

---

