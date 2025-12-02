# model_pipeline.py
import sqlite3
import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
DB_PATH = "building_damage.sqlite" # Placeholder for your SQLite file path
TARGET = "damage_grade"            # The column you are predicting
FEATURES = [
    'count_floors_pre_eq', 'age_building', 'area_percentage', 
    'height_percentage', 'land_surface_condition', 
    # Add all your final features here!
]
# Categorical features that need OneHotEncoding
OHE_FEATURES = ['land_surface_condition', 'foundation_type', 'roof_type']

# --- 2. DATA ACQUISITION & WRANGLING (SQL) ---
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

# --- 3. MODELING PIPELINE ---
def create_pipeline(model_type):
    """Creates a scikit-learn preprocessing and modeling pipeline."""
    # Define the preprocessing steps
    preprocessing = Pipeline(steps=[
        ('onehot', OneHotEncoder(use_cat_names=True, cols=OHE_FEATURES)),
        # Add other encoders or scalers if you used them!
    ])
    
    # Select the model based on input
    if model_type == 'logistic':
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    elif model_type == 'tree':
        model = DecisionTreeClassifier(max_depth=10, random_state=42) # Adjust max_depth
    else:
        raise ValueError("Invalid model_type specified.")
        
    # Combine preprocessor and model
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessing),
        ('classifier', model)
    ])
    return full_pipeline

# --- 4. VISUALIZATION (Odds Ratio Plot) ---
def plot_odds_ratios(log_reg_pipeline, feature_names, top_n=5):
    """Calculates and plots the odds ratio for the logistic regression model."""
    # *CRUCIAL: You must access the coefficients from the fitted pipeline*
    # This requires running the pipeline first and extracting the feature names after OHE
    
    # Example (Adjust based on your final model structure):
    # This part is highly customized and requires the fitted pipeline
    
    # ... Your code here to extract coefficients and corresponding feature names ...
    
    # Calculate Odds Ratios
    # odds_ratios = np.exp(coefficients) 
    
    # Create the plot using seaborn
    plt.figure(figsize=(10, 6))
    # sns.barplot(...) # Your specific plotting code
    plt.title(f'Top {top_n} Features: Odds Ratio for Building Damage')
    plt.ylabel('Odds Ratio')
    plt.axhline(1.0, color='red', linestyle='--') # Baseline
    plt.savefig('assets/odds_ratio_plot.png')
    plt.show()

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    data = connect_and_load_data(DB_PATH)
    if data is None:
        exit()
        
    X = data[FEATURES]
    y = data[TARGET]
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. Create and Train Pipeline
    log_reg_pipe = create_pipeline('logistic')
    print("Training Logistic Regression...")
    log_reg_pipe.fit(X_train, y_train)

    # 4. Evaluate
    train_acc = accuracy_score(y_train, log_reg_pipe.predict(X_train))
    test_acc = accuracy_score(y_test, log_reg_pipe.predict(X_test))
    
    print(f"\nFinal Training Accuracy: {train_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # 5. Visualize (Plotting the Odds Ratios)
    # The visualization code needs the fitted pipeline to extract coefficients
    # plot_odds_ratios(log_reg_pipe, FEATURES, top_n=5) # Run this line once the function is filled out
