
# 🧠 Text Analytics and Emotion Detection in Tweets

This project explores NLP techniques for emotion classification, topic modeling, named entity recognition (NER), and semantic similarity analysis—applied to tweets and biomedical text.

---

## 📌 Project Structure

### Task 1: Emotion Classification in Tweets
1. **Naive Bayes Classifier** – Traditional ML with bag-of-words features.
2. **CNN (Neural Network)** – Deep learning with pre-trained word embeddings.
3. **NMF Topic Modeling** – Discovering themes in tweets.

### Task 2: Biomedical Named Entity Recognition (NER)
4. **RoBERTa Sequence Tagger** – Transformer model for entity recognition.
5. **Disease Similarity** – Comparing diseases using BERT and Word2Vec embeddings.

---

## 🎯 Motivation

Understanding sentiment and extracting structured knowledge from unstructured biomedical or social media text is critical for applications in public health, social insight, and automated text systems. This project compares non-neural and neural NLP approaches in terms of performance and interpretability.

---

## 🧰 Tech Stack

- **Languages**: Python
- **Libraries**: `scikit-learn`, `nltk`, `spacy`, `tensorflow`, `keras`, `transformers`, `gensim`, `matplotlib`, `pandas`
- **Models**:
  - Naive Bayes
  - CNN (with GloVe embeddings)
  - RoBERTa (HuggingFace Transformers)
  - NMF
  - BERT & Word2Vec (Similarity analysis)

---

## 📂 Repository Files

| File | Description |
|------|-------------|
| `Naive_Bayes_1.1.ipynb` | Emotion classification using Naive Bayes |
| `Text_Analytics_1.2_Final_CNN.ipynb` | CNN-based tweet emotion classifier |
| `Text_Analytics_1.4_NMF.ipynb` | Topic modeling using NMF |
| `2.1.ipynb` | RoBERTa-based Named Entity Recognizer |
| `Text_Analytics_2.3.ipynb` | BERT & Word2Vec disease similarity |
| `Text_Analytics.pdf` | Full report with methodology, metrics & interpretation |

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/text-analytics-nlp.git
cd text-analytics-nlp
pip install -r requirements.txt
```

---

## 🚀 How to Run

- Run each notebook in order via Jupyter Notebook.
- Ensure that models requiring GloVe or RoBERTa are set up correctly (e.g., download weights if prompted).

---

## 📊 Evaluation & Results

### ✅ Naive Bayes (Multinomial)
- **Accuracy**: ~67%
- **F1 Scores** by emotion class:
  | Emotion   | Precision | Recall | F1-score | Support |
  |-----------|-----------|--------|----------|---------|
  | Anger     | 0.67      | 0.84   | 0.75     | 558     |
  | Joy       | 0.73      | 0.59   | 0.65     | 358     |
  | Optimism  | 0.51      | 0.26   | 0.34     | 123     |
  | Sadness   | 0.66      | 0.65   | 0.65     | 382     |

**Interpretation**:
- Strong performance on "Anger" and "Sadness" classes.
- Weaker generalization for minority classes like "Optimism".
- Suffers from class imbalance and feature independence assumption.

---

### ✅ CNN (Deep Learning Model with GloVe)
- **Test Accuracy**: 70.23%
- **F1 Score**: 0.63 (macro average)

**Interpretation**:
- CNN captures local contextual patterns better than Naive Bayes.
- Pre-trained embeddings boost understanding of word semantics.
- More robust to class imbalance but slightly overfits; can improve with regularization.

---

### ✅ Topic Modeling (NMF)
- **Preprocessing**: TF-IDF + tokenization + stopword removal.
- **Topics discovered** for *Joy* include:
  - 🎉 Celebrations: “birthday”, “happy”, “excited”
  - 🎬 Entertainment: “movie”, “hilarious”, “watched”
  - 👯‍♀️ Social Gratitude: “love”, “friend”, “thanks”

- For *Optimism*:
  - 💬 Motivational: “believe”, “blessed”, “truth”
  - 🧠 Mindset: “problem”, “overcome”, “leadership”

**Interpretation**:
- Topics align well with underlying emotions.
- Easy to interpret due to non-negativity constraint.
- Sensitive to number of topics and preprocessing quality.

---

### ✅ RoBERTa for Named Entity Recognition (NER)
- **F1 Scores** by entity:
  | Entity | Precision | Recall | F1 |
  |--------|-----------|--------|----|
  | Disease | 0.84     | 0.91   | 0.88 |
  | Chemical | 0.73    | 0.80   | 0.76 |
  | Anatomy | 0.57     | 0.85   | 0.68 |

**Interpretation**:
- RoBERTa shows excellent entity recognition, especially for disease-related tokens.
- Strong contextual embeddings capture complex entity spans.
- Minor issues: boundary mislabeling and over-prediction on ambiguous terms.

---

### ✅ Semantic Similarity (BERT & Word2Vec)
**Query Entity**: `dyskinesia`

**Top 5 Most Similar Diseases (BERT)**:
- `peptic` (0.77)
- `Hepatitis` (0.69)
- `arabinoside` (0.67)

**Least Similar (BERT)**:
- `retinoblastoma` (0.22)
- `menorrhagia` (0.24)

**Interpretation**:
- BERT captures nuanced semantic similarity between disease terms.
- Word2Vec yields less contextual but interpretable vector spaces.
- Useful for ontology design or clustering biomedical terms.

---

## 🔧 Limitations & Improvements

### For Naive Bayes:
- Handle class imbalance (e.g., SMOTE, class weighting)
- Use n-gram features or POS tags for richer inputs

### For CNN:
- Add dropout, early stopping to mitigate overfitting
- Explore BiLSTM or transformer-based classifiers

### For NER:
- Improve token alignment with subwords
- Add domain-specific lexicons or CRF layer

---

## 📈 Visuals

- 📉 Learning curves in CNN training indicate slight overfitting after 10 epochs.
- 📊 Bar charts of topic-word frequency per NMF cluster show distinct groupings.

---

## 🧑‍💻 Author

**Uchit Bhadauriya**  
MSc Data Science, University of Bristol  
🔗 [LinkedIn](https://www.linkedin.com/in/uchit-bhadauriya-a96478204)

---

## 📄 License

MIT License. See `LICENSE` for usage permissions.

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/)
- [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
- [Scikit-learn documentation](https://scikit-learn.org/)
- [Medium article on NMF](https://medium.com/@neri.vvo/non-negative-matrix-factorization-explained-practical-how-to-guide-in-python-c6372f2f6779)

---
