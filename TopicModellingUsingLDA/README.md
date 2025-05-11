# 📰 Topic Modeling with LDA on the National Public Radio Dataset

## 📌 Overview
This project applies **unsupervised topic modeling** using **Latent Dirichlet Allocation (LDA)** to group and analyze articles from the **National Public Radio (NPR)** dataset. The primary goal is to discover hidden thematic structures within a collection of news articles.

---

## 📂 Dataset
The dataset contains textual content from NPR articles. These texts were used as the input corpus for topic modeling.

---

## 🔍 Key Steps & Methodology

### 1. Data Exploration
- Loaded and inspected the NPR dataset.
- Viewed sample articles to understand content characteristics.

### 2. Text Preprocessing
- Tokenized the text data.
- Removed stopwords and non-alphabetic tokens.
- Applied lowercasing and basic normalization.

### 3. Vectorization
- Used `CountVectorizer` to convert text into a document-term matrix.
- Checked random tokens to evaluate vocabulary quality.

### 4. Topic Modeling with LDA
- Applied `LatentDirichletAllocation` from scikit-learn.
- Set the number of topics to **7** based on interpretability.
- Extracted top keywords for each topic.

### 5. Topic Inspection
- Interpreted the topics based on top words (e.g., politics-related topics).
- Reviewed the 7-topic output for semantic clarity.

### 6. Document-Topic Assignment
- Determined the dominant topic for each article.
- Combined topic labels with the original dataset for enriched analysis.

---

## 🧠 Tools & Libraries
- Python
- Pandas
- NumPy
- Scikit-learn (`CountVectorizer`, `LatentDirichletAllocation`)

---

## 📈 Output & Insights
- Articles were effectively grouped into coherent topics.
- Discovered meaningful themes within the dataset.
- Demonstrated the utility of LDA for unsupervised NLP tasks.

---

## 🔄 Future Work
- Evaluate topic coherence and tune topic numbers.
- Visualize results using `pyLDAvis`.
- Explore dynamic topic modeling or neural topic models for improved results.

---
