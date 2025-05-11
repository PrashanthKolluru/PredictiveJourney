# 📝 Quora Questions Topic Modeling using Non-negative Matrix Factorization (NMF)

This project utilizes machine learning to identify and classify topics from a dataset containing questions from Quora. By employing Natural Language Processing (NLP) techniques and Non-negative Matrix Factorization (NMF), we uncover hidden patterns and topics within the questions.

---

## 📊 Process Overview

### 1️⃣ Data Import & Exploration
- **Loaded** the dataset `quora_questions.csv`, containing various user-generated questions from Quora.
- **Explored** the dataset to understand its structure and performed initial inspections to view random questions.

### 2️⃣ Data Preprocessing & Cleaning
- **Checked** for missing or duplicate entries and cleaned data to remove inconsistencies.
- **Performed** text normalization (lowercasing, punctuation removal, stopword removal) to prepare the data for vectorization.

### 3️⃣ Feature Extraction
- **Transformed** text data into numerical features using TF-IDF vectorization to highlight important terms.
- **Applied** text feature extraction to convert textual information into a suitable format for modeling.

### 4️⃣ Topic Modeling with NMF
- **Implemented** Non-negative Matrix Factorization (NMF) from scikit-learn to identify latent topics within the dataset.
- **Extracted** key terms associated with each identified topic to interpret and label them meaningfully.

### 5️⃣ Interpretation
- **Analyzed** the topics to clearly communicate discovered patterns.

---

## 📈 Project Highlights

| Metric/Method | Description |
|---------------|-------------|
| Technique     | Non-negative Matrix Factorization (NMF) |
| Vectorizer    | TF-IDF |
| Topics        | Identified and clearly labeled latent topics with selected dataset |
| Insights      | Key terms clearly represent meaningful themes |

---

## 🔍 Insights from Topics

- **Technology**: Topics included questions about Laptops, Mobile purchases, EMI.
- **Politics**: Detected queries related to elections, politicians, president, prime minister.
- **Programming**: Questions about learning programming language and queries related ML and AI.

---

## 🔥 Conclusion

- ✅ Successfully identified meaningful and coherent topics from Quora questions.
- 📊 Leveraged NLP techniques combined with NMF for efficient topic modeling.
- 🧠 Gained insights valuable for content categorization and recommendation systems.

---

## 🚧 Future Improvements

- Enhance preprocessing techniques with lemmatization and part-of-speech tagging.
- Integrate the model into an application or website for real-time topic recommendations.

---

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn (NMF, TF-IDF Vectorizer)
- Natural Language Processing (NLP)

---

## 📁 Dataset

- Dataset `quora_questions.csv` contains thousands of user-submitted questions from Quora, suitable for NLP-based topic modeling.
