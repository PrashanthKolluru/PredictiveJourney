# Diabetes Prediction using Machine Learning 🚀

This project aims to predict diabetes in patients using machine learning algorithms. The dataset used contains health information from **768 individuals**, with **268 diabetic** and **500 non-diabetic** cases. The model is trained using a Support Vector Machine (SVM) classifier with standardized input features for optimal accuracy.

---

## 📊 Process Overview

### 1️⃣ Data Import & Exploration
- Imported necessary libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
- Loaded the dataset and analyzed its structure.
- Dataset contains **768 rows** and **9 columns** (8 features + target variable `"Outcome"`).
- Checked for missing values (**none found**).

---

### 2️⃣ Data Preprocessing & Feature Scaling
- Standardized features using `StandardScaler` to enhance the performance of the SVM model.
- Ensures all features have zero mean and unit variance.

---

### 3️⃣ Splitting Data into Training & Testing
- Features (X) and target variable (Y) were separated.
- Data was split into **80% training set** (614 samples) and **20% test set** (154 samples).
- Stratified sampling used to maintain the proportion of diabetic and non-diabetic cases.

---

### 4️⃣ Model Training
- Used `SVC(kernel='linear')` from `scikit-learn` for training.
- The model was trained on the standardized training data.

---

### 5️⃣ Model Evaluation
- Evaluated the model using accuracy score and confusion matrix.
- **Training Accuracy:** 78.66%
- **Test Accuracy:** 72.73%

---

## 📈 Model Performance

| Dataset      | Accuracy |
|--------------|----------|
| Training Set | 78.66%   |
| Test Set     | 72.73%   |

The SVM model shows decent predictive capability. However, further optimization can be done using hyperparameter tuning, ensemble techniques, or other advanced models.

---

## 🔍 Example Prediction

Tested the model with the following input:

```python
(1, 103, 30, 38, 83, 43.3, 0.183, 33)
```
✅ Prediction Result: The person is not diabetic

## 🔥 Conclusion


- ✅ Built a machine learning model using Support Vector Machine (SVM) for diabetes prediction.

- 📊 Achieved 78.66% training accuracy and 72.73% testing accuracy.

- 🔍 The dataset was clean (no missing values), and proper feature scaling was applied.

- 🛠️ The model successfully predicts outcomes based on health parameters like glucose, BMI, insulin, etc.

## 🚧 Future improvements could involve:

- Hyperparameter tuning

- Trying advanced models like Random Forest, XGBoost

- Applying techniques like SMOTE for class imbalance

- Deployment using Flask or Streamlit for real-time prediction

This project demonstrates how machine learning can be effectively used to predict diabetes and support medical decision-making. 🚀


