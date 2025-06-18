# EEE486 - Statistical Foundations of Natural Language Processing

This repository contains assignment implementations for the **EEE486** course on **Statistical Foundations of NLP**. The projects explore essential NLP tasks such as **collocation discovery**, **textual entailment classification**, and **dialogue summarization** — using both statistical and neural network-based methods.

---

## 📦 Assignments

### 🔹 Assignment 1 – The Hunt for Collocations | *Python, NLTK, NumPy*

- Preprocessed raw text using **tokenization**, **POS tagging**, and **lemmatization**
- Extracted **bigrams** based on syntactic patterns like *Adjective-Noun* and *Noun-Noun*
- Evaluated statistical significance using:
  - **Student’s t-test**
  - **Chi-squared test**
  - **Log-likelihood ratio test**
- Compared collocations under different **window sizes** and **significance thresholds** (α = 0.005)
- Provided linguistic analysis of top-ranked word pairs

> 📄 [View Report (PDF)](./The%20Hunt%20for%20Collocations/report_AltintasBerkay22002709.pdf)

---

### 🔹 Assignment 2 – Fine-Tuning BERT for Text Classification | *Python, Hugging Face Transformers*

- Fine-tuned `bert-base-uncased` on the **Recognizing Textual Entailment (RTE)** task from the GLUE benchmark
- Compared three pooling strategies:
  - `[CLS]` token representation (baseline, best-performing)
  - **Max pooling** over token embeddings
  - **Attention-based pooling** using trainable weights
- Performed extensive **hyperparameter tuning** (learning rate, max length, batch size, dropout)
- Reported results:
  - Best accuracy with `[CLS]` pooling: **67.51%**
  - Max pooling: **61.38%**
  - Attention pooling: **56.46%**

> 🔗 [View Model on Hugging Face](https://huggingface.co/berkayaltntas/bert-base-uncased-finetuned-rte-run_3)  
> 📄 [View Report (PDF)](./Fine-tuning%20BERT%20for%20Text%20Classification/report_AltintasBerkay22002709.pdf)

---

### 🔹 Assignment 3 – Implementing Transformer Architecture for Dialogue Summarization | *Python, TensorFlow, NumPy*

- Implemented a full **Transformer encoder-decoder model from scratch** for abstractive dialogue summarization
- Trained the model using a **custom attention-based architecture** with:
  - Scaled dot-product attention
  - Multi-head attention
  - Positional encoding
- Used a **paired dialogue-summary dataset** and trained for 300 epochs
- Evaluated using **BERTScore** with `roberta-large` to compare generated summaries with human references
- Presented both qualitative (example summaries) and quantitative (F1 scores) evaluation

> 📓 [View Code (Notebook)](./Implementing%20Transformer%20Architecture%20for%20Dialogue%20Summarization/code_AltintasBerkay22002709.ipynb)  
> 📄 [View Report (PDF)](./Implementing%20Transformer%20Architecture%20for%20Dialogue%20Summarization/report_AltintasBerkay22002709.pdf)
