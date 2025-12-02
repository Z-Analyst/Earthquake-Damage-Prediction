import sqlite3
import pandas as pd
import numpy as np
from category_encoders import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.validation import check_is_fitted

# --- 1. CONFIGURATION ---
DB_PATH = "building_damage.sqlite" # <--- CUSTOMIZE: Your SQLite file path
TARGET = "damage_grade"            # The column you are predicting
# <--- CUSTOMIZE: List all features used in your model, including categorical ones
FEATURES = [
    'count_floors_pre_eq', 'age_building', 'area_percentage', 
    'height_percentage', 'land_surface_condition', 'foundation_type', 
    'roof_type', 'has_secondary_use_gov' # Add all your final features!
]
# <--- CUSTOMIZE: List only the categorical features that need OneHotEncoding
OHE_FEATURES = ['land_surface_condition', 'foundation_type', 'roof_type']

# --- 2. DATA ACQUISITION & SPLITTING ---
def connect_and_load_data(db_path, query="SELECT * FROM main_table;"):
    """Connects to SQLite and loads data."""
    print("Connecting to SQLite database...")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def split_data(df, features, target):
    """Performs 3-way split: Training, Validation (for tuning), and Test."""
    X = df[features]
    y = df[target]
    
    # First split: Train (80%) vs Temp (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Second split: Validation (50% of temp) vs Test (50% of temp) => 10% each
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    # Combining Train and Validation for GridSearchCV to use cross-validation
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    
    return X_train_val, X_test, y_train_val, y_test

# --- 3. PIPELINE SETUP AND TUNING ---
def create_tuning_pipeline():
    """Defines the preprocessing and Decision Tree base structure."""
    preprocessing = Pipeline(steps=[
        ('onehot', OneHotEncoder(use_cat_names=True, cols=OHE_FEATURES)),
        # Add other encoders or scalers here if you used them!
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessing),
        # Decision Tree is the classifier we will tune
        ('classifier', DecisionTreeClassifier(random_state=42)) 
    ])
    return pipeline

def tune_and_fit_model(pipeline, X_train_val, y_train_val):
    """Uses GridSearchCV to find the best hyperparameters."""
    print("\nStarting Hyperparameter Tuning...")
    
    # <--- CUSTOMIZE: Define the hyperparameter grid you used
    param_grid = {
        'classifier__max_depth': range(5, 15, 2), # e.g., depths 5, 7, 9, 11, 13
        'classifier__min_samples_leaf': [5, 10, 20] 
    }
    
    # Use cross-validation (cv=5) on the training/validation set
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring='accuracy', # Use the metric you optimized for
        cv=5, 
        verbose=1,
        n_jobs=-1
    )
    
    search.fit(X_train_val, y_train_val)
    
    print(f"Best parameters found: {search.best_params_}")
    print(f"Best cross-validation accuracy: {search.best_score_:.4f}")
    
    # Return the best fitted model pipeline
    return search.best_estimator_

# --- 4. VISUALIZATION (Gini Importance) ---
def plot_gini_importance(tree_pipeline, X_data, top_n=10):
    """Calculates and plots the top N Gini Importance features."""
    
    # 1. Get the fitted Decision Tree model and preprocessor
    tree_model = tree_pipeline.named_steps['classifier']
    preprocessor = tree_pipeline.named_steps['preprocessor']
    
    # 2. Get the feature names after OneHotEncoding
    # Fit the preprocessor separately to X_data to ensure feature name mapping is correct
    preprocessor.fit(X_data) 
    final_features = preprocessor.get_feature_names_out(X_data.columns)
    
    # 3. Get importances and create DataFrame
    importances = tree_model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': final_features,
        'Importance': importances
    })
    
    # Remove features with zero importance (if any) and sort
    feature_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0]
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    # 4. Create the plot 
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title(f'Top {top_n} Feature Gini Importance for Damage Prediction')
    plt.xlabel('Gini Importance')
    plt.ylabel('Feature')
    
    # Save the file to the assets folder as planned
    import os
    if not os.path.exists('assets'):
        os.makedirs('assets')
        
    plt.savefig('assets/gini_importance_plot.png')
    plt.show()

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    data = connect_and_load_data(DB_PATH)
    if data is None:
        exit()
        
    # 2. Split Data
    # X_train_val is used for the GridSearchCV tuning; X_test is for final evaluation
    X_train_val, X_test, y_train_val, y_test = split_data(data, FEATURES, TARGET)

    # 3. Create and Tune Pipeline
    base_pipeline = create_tuning_pipeline()
    best_dt_pipeline = tune_and_fit_model(base_pipeline, X_train_val, y_train_val)

    # 4. Evaluate Final Model on Test Set
    train_acc = accuracy_score(y_train_val, best_dt_pipeline.predict(X_train_val))
    test_acc = accuracy_score(y_test, best_dt_pipeline.predict(X_test))
    
    print("\n--- Final Results ---")
    print(f"Final Training Accuracy: {train_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # 5. Visualize Feature Importance
    plot_gini_importance(best_dt_pipeline, X_train_val, top_n=10)
