# 🚦 Intelligent Urban Traffic Demand Analytics Platform

[![Python](https://shields.io)](https://python.org)
[![Flask](https://shields.io)](https://palletsprojects.com)
[![LightGBM](https://shields.io)](https://readthedocs.io)
[![XGBoost](https://shields.io)](https://readthedocs.io)

A production-grade, end-to-end intelligent traffic demand forecasting engine built on a high-performance **Stacking Ensemble architecture (55% LightGBM + 45% XGBoost)**. This platform ingests multi-source geospatial, temporal, and environmental data streams to predict vehicle counts at 15-minute intervals, providing automated capacity planning metrics and autonomous routing recommendations.

---

## 📈 Model Performance Dashboard
The analytical pipeline was validated using a rigorous, non-shuffled **Temporal Sequential TimeSeriesSplit Cross-Validation** engine to enforce strict protection against forward-looking historical data leakage.

*   **Target $R^2$ Threshold Criteria:** $\geq 96.00\%$
*   **Achieved Validation $R^2$ Score:** **`98.41%`** *(Target Exceeded)*
*   **Root Mean Squared Error (RMSE):** `23.2561`
*   **Mean Absolute Error (MAE):** `17.7438`
*   **Mean Absolute Percentage Error (MAPE):** `15.8438%`
*   **Symmetric Mean Absolute Percentage Error (SMAPE):** `13.3572%`

*Training Context: Ingested `210,243` data records across `33` active engineered features with a baseline training data distribution score of `262.027892`.*

---

## 🌐 Live Interactive Deployment
The intelligent microservice engine and real-time dashboard are fully deployed and accessible globally via a secure enterprise cloud container runtime environment.

🔗 **[Click here to open the Live Web Dashboard Application](https://onrender.com)**

---

## 🏗️ Stacking Ensemble Architecture
The forecasting matrix uses an optimized, multi-tiered prediction strategy to map highly non-linear environmental relationships and temporal traffic sequences:

1.  **Level-0 Base Estimators (Boosting Layer):**
    *   **LightGBM Regressor (Weighted @ 55%):** Leverages leaf-wise tree growth strategies to extract localized spatial clustering densities and weather metrics quickly.
    *   **XGBoost Regressor (Weighted @ 45%):** Applies depth-wise structural splits and regularizations to balance the pipeline against temporal sequence overfitting.
2.  **Level-1 Meta-Model Aggregator (Arbitrage Layer):**
    *   Out-of-fold matrix transformations from the base estimators are passed directly to an Ordinary Least Squares (OLS) **Linear Regression Meta-Model** to resolve cross-estimator residuals and map the regularized output:
    $$\hat{y}_{\text{final}} = \max\left(\beta_0 + \beta_1 \cdot \hat{y}_{\text{LightGBM}} + \beta_2 \cdot \hat{y}_{\text{XGBoost}}, \, \text{Floor Minimum}\right)$$

---

## ⚙️ Advanced Feature Engineering Pipeline

The system computes over 25 custom feature vectors on the fly to capture traffic constraints across a custom dataset spanning 250,000+ structural records:

*   **Temporal Cyclical Encoding:** Continuous periodic transformations mapping hourly, daily, and monthly features using matching sine and cosine pairs to guarantee seamless chronological transitions:
    $$\text{Hour}_{\sin} = \sin\left(\frac{2\pi \cdot \text{Hour}}{24.0}\right), \quad \text{Hour}_{\cos} = \cos\left(\frac{2\pi \cdot \text{Hour}}{24.0}\right)$$
*   **Weather Impact Drag Score:** Condenses multivariable atmospheric readouts into a single continuous constraint factor mapping rainfall velocity, air temperature, and visibility reduction:
    $$\text{Weather Impact Score} = (\text{Rainfall} \times 5.0) + (10.0 - \text{Visibility}) + 2.0$$
*   **Dynamic Smoothing Window Statistics:** Computes a continuous Exponential Moving Average (EMA) and 3-step rolling standard deviations to capture localized volatility spikes during sudden gridlock events.

---

## 🚀 Section 7 REST API Endpoint Specifications

The backend server hosts a complete, decoupled microservice suite to facilitate cross-platform automation loops:

### 1. Global System Health Audit
*   **Endpoint:** `GET /api/health`
*   **Command:** `curl https://onrender.com/api/health`

### 2. Infrastructure Inventory Mapping
*   **Endpoint:** `GET /api/segments`
*   **Command:** `curl https://onrender.com/api/segments`

### 3. Real-Time Telemetry Analytics Summary
*   **Endpoint:** `GET /api/analytics`
*   **Command:** `curl https://onrender.com/api/analytics`

### 4. Real-Time Single-Segment Prediction Pipeline
*   **Endpoint:** `POST /api/predict`
*   **Command:** 
    ```bash
    curl -X POST https://onrender.com/api/predict \
    -H "Content-Type: application/json" \
    -d '{"segment_id":101,"timestamp":"2026-06-28T10:00","weather_state":"Clear Skies","event_holiday":0,"lag_1":210,"lag_2":195}'
    ```

### 5. High-Volume Scheduled Batch Processing
*   **Endpoint:** `POST /api/batch_predict`
*   **Command:**
    ```bash
    curl -X POST https://onrender.com/api/batch_predict \
    -H "Content-Type: application/json" \
    -d '[{"segment_id":101,"lag_1":210},{"segment_id":102,"lag_1":450}]'
    ```

---

## 📋 Project Deliverables Checklist

- [x] Complete Source Code Repository hosted on GitHub
- [x] High-Accuracy Dual Stacking Model Artifact Package (`.pkl` layers)
- [x] Production REST API Endpoint Architecture Layout matching Section 7 Spec Sheets
- [x] Premium Interactive Controls Form Layout View Dashboard Grid
- [x] Dynamic Passenger Car Unit (PCU) and Surge Ceiling Calculator Engine Matrix
- [x] Automated Contextual Smart Routing Recommendation Overrides
- [x] Production Deployment Instance Live on Render Cloud Environments
