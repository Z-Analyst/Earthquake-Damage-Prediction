# Earthquake-Damage-Prediction
# 🏚️ Nepal 2015 Earthquake Damage Prediction

## 💡 Project Goal
Developed a machine learning classification model to **predict the severity of damage** sustained by buildings during the 2015 Nepal Earthquake. The goal was to identify the most vulnerable structures based on pre-earthquake features to assist with future disaster relief prioritization and structural risk assessment.

## 📊 Key Results
The final classification model significantly improved predictive accuracy over the baseline, offering a viable method for initial damage assessment.
Logistic Regression model used
* **Baseline Accuracy (Majority Class):** `0.64`
* **Training Accuracy:** `0.71`
* **Test Accuracy:** `0.72`
    * *Result:* The model achieved an accuracy improvement of approximately **12.5%** over the naive baseline, indicating strong feature selection and model fit for this multi-class classification problem.

## 🛠️ Methodology and Techniques

This project involved a comprehensive machine learning pipeline, emphasizing data handling and model interpretation:

### 1. Data Wrangling & Preparation
* **Source:** Data was sourced from **Open Data Nepal**.
* **Database:** Used **SQL / SQLite** for efficient data extraction, transformation, and querying of the relational dataset, demonstrating proficiency in database management for ML projects.
* **Encoding:** Applied both **OneHotEncoder** and **OrdinalEncoder** to handle categorical features within the scikit-learn pipeline.

### 2. Comparative Classification Modeling
* **Models Explored:**
    * **Logistic Regression:** Used as an interpretable baseline model. Feature coefficients were analyzed to understand the *odds ratio* of different building attributes contributing to damage.
    * **Decision Tree Classifier:** Employed to capture non-linear relationships and interactions between features, providing a different perspective on feature importance.
* **Pipeline:** Implemented a **scikit-learn Pipeline** to streamline preprocessing (encoding) and modeling, ensuring consistency and preventing data leakage.

### 3. Ethical Considerations
* Incorporated ethical considerations into the model building process, specifically focusing on **fairness** (ensuring predictions did not disproportionately misclassify buildings based on socio-economic proxies) and **interpretability** (using feature importance and odds ratios to communicate risk transparently to stakeholders).

## 📈 Visual Insight

The most insightful visualization involved plotting the **Odds Ratios for the five largest features** derived from the Logistic Regression model. This plot clearly communicated *which* building characteristics (e.g., roof type, foundation, age) most dramatically increased or decreased the likelihood of damage, offering actionable intelligence.

## ⚙️ Repository Structure

* `model_pipeline.py`: Clean, modularized Python code containing all functions for SQL connection, data loading, preprocessing, modeling, and evaluation.
* `requirements.txt`: A list of necessary libraries.
* `assets/`: Folder containing static images of key visualizations.
