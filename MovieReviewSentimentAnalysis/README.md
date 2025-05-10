# 🎬 Movie Review Sentiment Analysis using Machine Learning

This project focuses on classifying movie reviews as either positive or negative using machine learning techniques. The model leverages Natural Language Processing (NLP) with TF-IDF vectorization and a Linear Support Vector Machine (SVM) for sentiment classification.

---

## 📊 Process Overview

### 1️⃣ Data Import & Cleaning
- Loaded a tab-separated dataset (`moviereviews2.tsv`) containing movie reviews and their associated sentiment labels (`pos` or `neg`).
- Checked for null and empty values; removed entries with missing or blank reviews.

### 2️⃣ Feature Extraction
- Converted the text data into numerical format using **TF-IDF Vectorizer** from `scikit-learn`.

### 3️⃣ Data Splitting
- Features (`X`) and labels (`y`) were separated.
- Dataset was split into 70% training and 30% testing sets using `train_test_split`.

### 4️⃣ Model Building & Training
- Created a `Pipeline` combining `TfidfVectorizer` and `LinearSVC`.
- Trained the model on the preprocessed training data.

### 5️⃣ Model Evaluation
- Evaluated using **accuracy score**, **confusion matrix**, and **classification report**.
- The model achieved high accuracy on the test set and performed well in distinguishing positive and negative sentiments.

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Classifier | LinearSVC |
| Vectorizer | TF-IDF |
| Accuracy | 0.93756 or 93.756% ≈ 94% |

---

## 🔍 Example Prediction

The model was tested on real-world samples from the dataset and predicted sentiment as either:

✅ `positive`  
❌ `negative`

Model predicted the sample review "This movie was a waste of time and money spent!" as ❌ `negative`.  
Model predicted the sample review "This movie was super good and I enjoyed thoroughly!" as ✅ `positive`.


---

## 🔥 Conclusion

✅ Built a text classification model using TF-IDF and Support Vector Machine (SVM).  
📊 Applied preprocessing techniques to clean and vectorize text data.  
🧠 Achieved strong sentiment classification accuracy.  
🚀 The model can be extended for real-time sentiment analysis in apps or websites.

---

## 🚧 Future Improvements

- Hyperparameter tuning for better performance
- Experimenting with alternative models like **Logistic Regression**, **Naive Bayes**, or **Random Forest**
- Enhancing with **word embeddings** (e.g., Word2Vec, GloVe)
- Building a user interface with **Streamlit** or **Flask** for live predictions

---

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn
- Natural Language Processing (NLP)
- Machine Learning (SVM)

---

## 📁 Dataset

The dataset used in this project is a TSV file of movie reviews labeled as `positive` or `negative`.

---
