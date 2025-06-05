# Emotion and Entity Analysis using NLP Techniques

This repository contains a comprehensive Natural Language Processing (NLP) project focused on emotion classification, topic modeling, named entity recognition (NER), and semantic similarity analysis using state-of-the-art models such as Naive Bayes, CNN, RoBERTa, and BERT/Word2Vec embeddings.

## Project Structure

```
├── Naive_Bayes_1.1.ipynb
├── Text_Analytics_1.2_Final_CNN.ipynb
├── Text_Analytics_1.4_NMF.ipynb
├── 2.1.ipynb
├── Text_Analytics_2.3.ipynb
├── Text_Analytics.pdf
├── images/
│   └── output_*.png
├── README.md
```

## Tasks Overview

### Task 1: Emotion Classification

#### 🔹 Naive Bayes
- Preprocessing includes tokenization, stopword removal, lemmatization, and count vectorization.
- Baseline model achieving ~67% accuracy.

#### 🔹 CNN Model
- Used GloVe embeddings and convolutional layers.
- Accuracy ~70.2%, F1-score 0.63.

### Task 2: Named Entity Recognition (NER)

#### 🔹 RoBERTa
- BIO tagging for biomedical entities.
- Strong precision and recall observed.

### Task 3: Topic Modeling

#### 🔹 NMF
- Revealed interpretable themes for joy and optimism using TF-IDF and latent topics.

### Task 4: Entity Similarity

#### 🔹 BERT vs Word2Vec
- BERT embeddings were semantically richer.
- Word2Vec was faster but less contextual.

---

## 📊 Visual Output

<p align="center"><img src="images/output_1.png" width="500"/></p>
<p align="center"><img src="images/output_10.png" width="500"/></p>
<p align="center"><img src="images/output_11.png" width="500"/></p>
<p align="center"><img src="images/output_12.png" width="500"/></p>
<p align="center"><img src="images/output_2.png" width="500"/></p>
<p align="center"><img src="images/output_3.png" width="500"/></p>
<p align="center"><img src="images/output_4.png" width="500"/></p>
<p align="center"><img src="images/output_5.png" width="500"/></p>
<p align="center"><img src="images/output_6.png" width="500"/></p>
<p align="center"><img src="images/output_7.png" width="500"/></p>
<p align="center"><img src="images/output_8.png" width="500"/></p>
<p align="center"><img src="images/output_9.png" width="500"/></p>
<p align="center"><img src="images/word_graph_1.png" width="500"/></p>
<p align="center"><img src="images/word_graph_10.png" width="500"/></p>
<p align="center"><img src="images/word_graph_11.png" width="500"/></p>
<p align="center"><img src="images/word_graph_12.png" width="500"/></p>
<p align="center"><img src="images/word_graph_2.png" width="500"/></p>
<p align="center"><img src="images/word_graph_3.png" width="500"/></p>
<p align="center"><img src="images/word_graph_4.png" width="500"/></p>
<p align="center"><img src="images/word_graph_5.png" width="500"/></p>
<p align="center"><img src="images/word_graph_6.png" width="500"/></p>
<p align="center"><img src="images/word_graph_7.png" width="500"/></p>
<p align="center"><img src="images/word_graph_8.png" width="500"/></p>
<p align="center"><img src="images/word_graph_9.png" width="500"/></p>

---

## Usage

```bash
python main.py
```

Or run notebooks sequentially for each task in Jupyter. Ensure all data files are available locally.

## References

- VDJdb: Goncharov et al. (2022)
- TCRdist3: https://tcrdist3.readthedocs.io
- UMAP: Becht et al. (2019)
- Additional: IBM, Medium, Wikipedia, TowardsDataScience

## Contact

**Created by:** Uchit Bhadauriya  
**Institution:** University of Bristol  
📫 ir23063@bristol.ac.uk  
🔗 [LinkedIn](https://www.linkedin.com/in/uchit-bhadauriya-a96478204)

⭐ Star this repo if you found it useful!  
🛠️ Fork it to build on this work  
❓ Open an issue for questions or collaboration
