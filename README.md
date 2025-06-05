
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

## 📊 Evaluation & Results (with Visuals)

### ✅ Naive Bayes (Multinomial)

- **Test Accuracy**: `67%`
- **Macro F1 Score**: `0.60`

**Test Set Classification Report:**

![Naive Bayes Classification Report](./Screenshot%202025-06-05%20at%2011.14.45%20AM_resized.png)

> 📌 Naive Bayes performs well for the "Anger" class (F1 = 0.75), but underperforms on minority classes like "Optimism", highlighting sensitivity to class imbalance.

---

### ✅ CNN (Deep Learning Model with GloVe)

- **Test Accuracy**: `70.23%`
- **F1 Score**: `0.6295`

**Training vs. Validation Loss Curve:**

![CNN Training Curve](./Screenshot%202025-06-05%20at%2011.14.57%20AM_resized.png)

> 📉 The validation loss begins to plateau after epoch 7, suggesting mild overfitting. Early stopping could be considered to improve generalization.

---

### 📈 Topic Modeling: NMF

**Class Distribution of Emotions:**

![Emotion Frequency Bar Chart](./Screenshot%202025-06-05%20at%2011.15.10%20AM_resized.png)

##### 🔹 *Top 5 Joy Topics (NMF) with Word Clouds:*

- **Topic 1 – Media & Events**
![Topic 1 - Joy](./Screenshot%202025-06-05%20at%2011.15.16%20AM_resized.png)

- **Topic 2 – Humor & Movies**
![Topic 2 - Joy](./Screenshot%202025-06-05%20at%2011.16.47%20AM_resized.png)

- **Topic 3 – Celebration & Emotions**
![Topic 3 - Joy](./Screenshot%202025-06-05%20at%2011.16.53%20AM_resized.png)

- **Topic 4 – Friendship & Gratitude**
![Topic 4 - Joy](./Screenshot%202025-06-05%20at%2011.17.00%20AM_resized.png)

- **Topic 5 – Fun & Social Interaction**
![Topic 5 - Joy](./Screenshot%202025-06-05%20at%2011.17.06%20AM_resized.png)

---

##### 🔹 *Optimism Themes (NMF) with Word Cloud:*

- **Topic 1 – Leadership, Worry, Resilience**

![Topic 1 - Optimism](./Screenshot%202025-06-05%20at%2011.17.12%20AM_resized.png)

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
