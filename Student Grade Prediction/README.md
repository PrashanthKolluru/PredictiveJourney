# 🎓 Student Grade Prediction Using Machine Learning 📈

Welcome to this exciting Machine Learning project, where I've explored predictive analytics to forecast student grades using detailed academic and socio-economic data. The goal is clear: **Build robust ML models to reliably predict student performance**. Such predictions provide insights for timely interventions, ultimately improving educational outcomes.

---

## 🚀 Project Overview
This project leverages a comprehensive dataset from high school students in Montreal, focusing on predicting three critical performance indicators:

- **G1** *(First period grade)*
- **G2** *(Second period grade)*
- **G3** *(Final grade)*

### 📌 Key Features Included:
- 📚 **Study Time**
- 🚌 **Travel Time**
- 🏠 **Socio-Economic Background** *(Parental education, Job status, Address, etc.)*
- 💡 **Behavioral Factors** *(Health status, Absences, Alcohol consumption behaviour of parents, Social activities, etc.)*

---

## 📂 Data & Preprocessing
The dataset underwent extensive preprocessing:

- ✅ Verified absence of missing data.
- 🧹 Categorical variables were encoded appropriately.
- 🔄 Normalization of numerical features for improved model accuracy.

---

## 📊 Exploratory Data Analysis (EDA)
Insights gathered include:

- **Grades Distribution**:
  - Grades (G1, G2, G3) follow bell-shaped distributions, indicating consistency in student performance patterns.
- **Correlation Analysis**:
  - Strong correlation among grades (0.8-0.9), highlighting interdependence.
  - Significant predictors include "Absences," "Failures," "Parental Education," and "Study Time."
- **Chi-Square Tests**:
  - "Absences," "Failures," and "Alcohol Consumption" emerged as significant factors influencing performance.

---

## 🛠️ Modeling & Evaluation
Two ML approaches were applied: **Classification** & **Regression**.

### 🔸 Classification Models (Random Forest)
- **G1** Prediction Accuracy: **71%**
- **G2** Prediction Accuracy: **70%**
- **G3** Prediction Accuracy: **81%** 🎉 *(Excellent predictive capability!)*

**Key Predictors**:
- "Absences," "Health," "Social Outings," "Age."

### 🔹 Regression Models
Models predicting exact grades yielded impressive results:

| Grade | R² Score | Mean Absolute Error (MAE) |
|-------|----------|---------------------------|
| **G1**   | 0.80     | 1.31                      |
| **G2**   | 0.78     | 1.45                      |
| **G3**   | 0.83 🎯  | 1.09                      |

**Top Influential Features**:
- "Failures," "Absences," "Mother's Job," "Study Time."

---

## 🌟 Key Insights & Takeaways
- 🚩 **Critical Factors**:
  - Student **Absences** and **Previous Failures** significantly influence academic outcomes.
- 👨‍👩‍👧‍👦 **Parental Influence**:
  - Socio-economic factors, including parental job and education levels, show considerable impact.
- 🧑‍🤝‍🧑 **Behavioral Aspects**:
  - Social activities, health, and alcohol consumption correlate notably with academic performance.

---

## 📌 Conclusion & Future Directions
This project highlights Machine Learning’s effectiveness in accurately predicting student grades, emphasizing the importance of academic, behavioral, and demographic factors.

🚧 **Future Improvements**:
- Expand dataset for broader generalizability.
- Address outliers through targeted preprocessing to further enhance model accuracy.

## 📧 Contact
Reach out if you have questions or suggestions regarding this project.
