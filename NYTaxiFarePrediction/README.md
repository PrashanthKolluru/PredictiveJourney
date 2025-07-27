# NYC Taxi Fare Prediction using a Tabular Neural Network

This project builds a regression model to accurately predict taxi fares in New York City based on trip distances, timestamps, and passenger counts. By leveraging embeddings for categorical time features and a deep feed‑forward network for continuous features, we achieve a robust fare estimator for real‑world ride‑hailing applications.

---

## 📊 Process Overview

1. **Data Import & Exploration**  
   - Loaded **120,000** taxi trip records from April 2010.  
   - Examined fare distribution (mean ≈ \$10.04, std ≈ \$7.50, min \$2.50, max \$49.90).

2. **Feature Engineering**  
   - Computed trip distance in kilometers using the **Haversine formula** between pickup and dropoff coordinates.  
   - Converted UTC timestamps to Eastern Daylight Time and extracted:
     - Hour of day  
     - AM/PM flag  
     - Weekday  
   - Separated features into:
     - **Categorical**: `Hour`, `AMorPM`, `Weekday`  
     - **Continuous**: `pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude`, `passenger_count`, `dist_km`

3. **Model Architecture**  
   - **Embeddings** for each categorical feature:  
     | Feature   | Categories | Embedding Size |
     |-----------|------------|----------------|
     | Hour      | 24         | 12             |
     | AM/PM     | 2          | 1              |
     | Weekday   | 7          | 4              |
   - **Continuous inputs** batch‑normalized and concatenated with embeddings.  
   - **Feed‑forward layers**: 200 → 100 units with ReLU, BatchNorm, and Dropout(p = 0.4).  
   - **Output layer**: single neuron for fare regression.

4. **Training & Evaluation**  
   - Optimizer: **Adam** (lr = 0.001)  
   - Loss: **RMSE** (converted from MSE)  
   - Trained for **300 epochs** on a **96,000‑record** split (20% held out).  
   - **Final Validation RMSE**: ± \$3.67

---

## 📈 Project Highlights

| Metric / Component      | Description                                         |
|-------------------------|-----------------------------------------------------|
| Dataset Size            | 120,000 records                                     |
| Fare Distribution       | mean ≈ \$10.04, std ≈ \$7.50, max \$49.90             |
| Technique               | Tabular Neural Network with Embeddings              |
| Categorical Embeddings  | [(24→12), (2→1), (7→4)]                             |
| Continuous Features     | 6 (coords, passenger count, distance)               |
| Hidden Layers           | [200, 100]                                          |
| Regularization          | BatchNorm & Dropout (p = 0.4)                       |
| Training Epochs         | 300                                                 |
| Optimizer               | Adam (lr = 0.001)                                   |
| Validation RMSE         | \$3.67                                              |

---

## 🔍 Model Insights

- **Distance** is the strongest continuous predictor—longer trips drive higher fares.  
- **Time features** (hour and weekday) capture surge patterns and demand fluctuations (e.g., late‑night rides tend to be pricier).  
- **Embedding layers** enable the network to learn nuanced relationships (weekday vs weekend pricing, AM vs PM).

---

## 🔥 Conclusion

- ✅ Developed a neural network that predicts NYC taxi fares with ∼\$3.70 RMSE on unseen data.  
- ✅ Demonstrated the power of combining embedding layers for categorical time features with continuous geospatial inputs.  
- ✅ Provided a production‑ready PyTorch model (`TaxiFareRegrModel.pt`) and an interactive prediction function.

---

## 🚧 Future Improvements

- Incorporate **weather data** (rain, snow) and **traffic conditions** to capture external fare drivers.  
- Experiment with **learning rate schedules** and **early stopping** to further reduce overfitting.  
- Use **geohash** or **clustered zones** instead of raw coordinates to enhance spatial encoding.  
- Scale up training on the full historical dataset for better generalization.

---

## 🛠️ Tech Stack

- **Python**  
- **pandas**, **NumPy**  
- **PyTorch**  
- **Matplotlib**  
- **Jupyter Notebook**

---

## 📁 Dataset

- **NYCTaxiFares.csv**: April 2010 NYC Yellow Taxi trip records with pickup/dropoff coordinates, timestamps, passenger counts, and fare amounts.
