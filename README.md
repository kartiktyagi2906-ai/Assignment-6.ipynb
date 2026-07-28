# AI-ML-Assignment-6
Weather Condition Classification using Support Vector Machine (SVM) and Open-Meteo API

Name: Kartik Tyagi

Registration Number: 23BAI10360

Application Number: IN26010566

Batch Number: 1A

Email ID: kartik.23bai10360@vitbhopal.ac.in

## Objective
The objective of this project is to build an SVM classification model using an RBF kernel to accurately classify weather conditions as 'Warm' ($\ge 25^\circ\text{C}$) or 'Cool' ($< 25^\circ\text{C}$) based on hourly meteorological features fetched live from the Open-Meteo API[cite: 2].

## Data Source & API Documentation
- **API Documentation:** [Open-Meteo API](https://open-meteo.com/)[cite: 2]
- **Endpoint Used:** Open-Meteo Historical Archive API (`temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`)[cite: 2]

## Libraries Used
- `requests`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Methodology
1. **Data Collection & Understanding**: Fetched historical hourly weather observations for Delhi (Lat: 28.61, Long: 77.23) and defined the target variable `Weather_Class`[cite: 2].
2. **Data Preprocessing**:
   - Dropped non-predictive attributes (`time`)[cite: 2].
   - Encoded binary target variable (`Warm`: 1, `Cool`: 0)[cite: 2].
   - Split dataset into 80% training and 20% testing sets using stratified sampling[cite: 2].
   - Standardized features using `StandardScaler` to equalize feature contributions across Euclidean distance margins[cite: 2].
3. **Model Development**: Trained a `SVC` model using the `rbf` kernel on standardized training features[cite: 2].
4. **Model Evaluation**: Evaluated performance using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix heatmap[cite: 2].

## Results

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 98.86% |
| **Precision** | 99.15% |
| **Recall** | 98.98% |
| **F1-Score** | 0.9906 |

## Conclusion
The SVM model ($RBF\text{ kernel}$) effectively separates warm and cool weather states using surface pressure, humidity, and wind speed features[cite: 2]. Feature standardization using `StandardScaler` is essential to prevent large-magnitude variables from biasing the optimal decision boundary margin[cite: 2]. While SVM provides strong non-linear decision boundary fitting, its computational cost scales heavily with dataset size[cite: 2].
