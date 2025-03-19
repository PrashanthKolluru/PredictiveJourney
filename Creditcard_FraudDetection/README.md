# **Credit Card Fraud Detection 🚀**

This project focuses on **detecting fraudulent transactions** using **machine learning** techniques. The dataset contains **284,807 transactions**, out of which only **492 are fraudulent**. Due to this extreme class imbalance, **under-sampling** is applied to balance the dataset before training the model.

---

## 📊 **Process Overview**

1️⃣ **Data Import & Exploration**  
- Loaded the dataset and checked its structure.  
- The dataset has **31 columns**, with the **"Class" column** as the target variable (`0 = Legitimate, 1 = Fraudulent`).  
- No missing values were found in the dataset.

2️⃣ **Handling Imbalanced Data**  
- The dataset is highly skewed, with **99.83% legitimate transactions** and only **0.17% fraudulent transactions**.  
- To balance this, **under-sampling** was used, reducing the number of legitimate transactions to match the fraudulent ones (`492 fraud + 492 legit`).  

3️⃣ **Feature & Target Splitting**  
- Features (`X`) and target variable (`Y`) were separated.  
- The data was **split into training (80%) and testing (20%) sets**.  

4️⃣ **Model Training**  
- **Logistic Regression** was used to train the model.  
- Encountered a **ConvergenceWarning**, which suggests that increasing `max_iter` or standardizing data could help.

5️⃣ **Model Evaluation**  
- The model was tested on both **training** and **testing datasets** to check accuracy.

---

## 📈 **Model Performance**

✅ **Training Accuracy:** `95.04%`  
✅ **Test Accuracy:** `90.35%`  

These results indicate that the model **performs well on both training and testing data**, meaning it can effectively differentiate between fraudulent and legitimate transactions.  

🔹 **However, since financial fraud detection is a high-risk problem, even a small misclassification can lead to serious consequences.** Further improvements, such as using **ensemble models** or **SMOTE for handling class imbalance**, can be explored to enhance the performance.

---

## 🔥 **Conclusion**

- The dataset was highly imbalanced, which was addressed using **under-sampling**.  
- A **Logistic Regression model** was trained and evaluated.  
- The model achieved a **90.35% accuracy** on the test data.  
- Although the accuracy is high, further improvements can be made using **advanced machine learning techniques** to minimize misclassification of fraud cases.

This project showcases how **machine learning can help detect fraudulent transactions**, providing valuable insights for financial security systems. 🚀
