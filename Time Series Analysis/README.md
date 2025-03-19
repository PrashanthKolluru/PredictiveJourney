# 📈 Time Series Analysis: Predicting Unique Customer Visits 🛒

Welcome to my Machine Learning project focused on **Time Series Analysis** to predict the number of unique customers visiting a local store in Montreal. By analyzing historical customer visit data, this project leverages powerful statistical models (**ARIMA & SARIMA**) to provide reliable forecasts, essential for strategic business planning and optimization.

---

## 🎯 Project Objective
The main goal of this project is to analyze online store visit patterns over time and accurately predict future unique visits. The insights from this analysis enable effective demand forecasting, resource planning, and targeted marketing strategies.

---

## 📊 Dataset Overview
The dataset includes the following key variables:

- 📅 **Date & Day**
- 🔄 **Page Loads**
- 👤 **First-Time Visits**
- ♻️ **Returning Visits**
- 🎯 **Target Variable**: Unique Visits *(Primary focus for prediction)*

---

## 🧹 Data Preprocessing
Data underwent several preprocessing steps:

- ✅ Checked and confirmed the absence of missing values.
- 🔢 Converted data types for uniformity (strings → integers → datetime).
- 📌 Date column was converted to datetime format and set as the dataset index.

---

## 📉 Exploratory Data Analysis (EDA)
Key insights from EDA included:

- **Seasonality in Customer Visits**: Strong seasonal trends with consistent peaks and valleys each year, aligned with holidays or promotional events.
- **Stable Long-term Trends**: Overall trend remains stable with minor fluctuations, suggesting steady but not rapidly increasing business performance.
- **Periodic Spikes and Troughs**: Clear periodic spikes indicate successful promotional activities or seasonal demand.

---

## 📈 Seasonality and Trend Decomposition
Using **Seasonal Decomposition**, I uncovered three main components:

- **Trend**: Clear long-term fluctuations indicating gradual shifts in customer visits over the years.
- **Seasonality**: Strong recurring patterns throughout the year.
- **Residual (Noise)**: Random fluctuations, indicating minimal unexplained variance.

---

## 🛠️ Modeling Techniques
Two primary models were used to forecast customer visits:

### 🔹 **ARIMA (Autoregressive Integrated Moving Average)**
- Captured overall trends and seasonality effectively.
- Data differencing transformed non-stationary data to stationary, suitable for ARIMA modeling.
- Provided solid short-term predictive power.

### 🔸 **SARIMA (Seasonal ARIMA)**
- Integrated seasonal components effectively, significantly improving predictions.
- Accurately modeled weekly seasonality patterns (with parameter configuration `SARIMA(2,1,2)(2,1,2,7)`).
- Demonstrated higher precision compared to standard ARIMA, particularly for seasonal data.

---

## 🚀 Model Evaluation & Accuracy
The **SARIMA** model demonstrated excellent forecasting performance:

- ✅ **In-Sample Forecast**: SARIMA closely tracked historical data with high accuracy.
- 🌟 **Near Future Predictions**: Forecasts showed strong alignment with historical seasonal patterns, maintaining accuracy.
- ⚠️ **Long-Term Predictions**: While maintaining seasonality, forecast accuracy gradually reduced, reflected by widening confidence intervals over extended periods.

**Model Diagnostics:**
- Residuals showed randomness with minimal autocorrelation, confirming the robustness of the SARIMA model.
- Residual distribution was approximately normal, indicating a well-fitted model without systematic bias.

---

## 🔮 Key Findings & Insights
- **Seasonality Dominates**: Customer visits heavily influenced by seasonal trends (holidays, events).
- **Stable Long-Term Performance**: No significant upward/downward long-term trend indicates consistent business.
- **Forecast Reliability**: SARIMA effectively captures recurring patterns,
