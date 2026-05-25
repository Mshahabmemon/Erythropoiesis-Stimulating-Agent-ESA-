# ============================================================
# AUTO INSTALL REQUIRED PACKAGES
# ============================================================
import subprocess
import sys
from pathlib import Path

def install_packages():
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'xgboost',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'openpyxl',
        'joblib'
    ]
    
    print("Checking and installing required packages...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")

install_packages()

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
from collections import Counter
import xgboost as xgb
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
MODEL_PATH = SCRIPT_DIR / "xgboost_model.pkl"
DATA_FILE = DATA_DIR / "Updated_Salman_Bhai_project_data.xlsx"

TARGET_COLUMN = "New ESA Dose"
TEST_SIZE = 0.2
RANDOM_STATE = 42

COLUMNS_TO_DROP = [
    'Unnamed: 31',
    'Unnamed: 0',
    'Unnamed: 2',
    'New ESA Dose2',
    'New Dose2 Effective Date',
    'Effective Date.1',
    'Dose Change_numeric',
    'New Dose Effective Date_numeric'
]


# ============================================================
# DATA CLEANING FUNCTIONS
# ============================================================

def detect_dominant_type(series):
    types = series.dropna().apply(lambda x: type(x)).tolist()
    if not types:
        return None
    return Counter(types).most_common(1)[0][0]


def coerce_to_dominant_type(series):
    dominant_type = detect_dominant_type(series)

    if dominant_type in [int, float, np.int64, np.float64]:
        return pd.to_numeric(series, errors="coerce")

    if dominant_type == str:
        return series.astype(str)

    return series


def extract_numeric_from_text(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return value

    match = re.search(r'(\d+\.?\d*)', str(value))
    if match:
        return float(match.group(1))

    return np.nan


def parse_dates(series):
    return pd.to_datetime(series, errors="coerce")


def clean_data(file_path):
    print("Loading data...")
    df = pd.read_excel(file_path)

    # Drop columns safely
    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns],
                 errors="ignore")

    # Coerce dominant types
    for col in df.columns:
        df[col] = coerce_to_dominant_type(df[col])

    # Extract dose values automatically
    for col in df.columns:
        if "dose" in col.lower():
            df[col] = df[col].apply(extract_numeric_from_text)

    # Parse date columns
    for col in df.columns:
        if "date" in col.lower():
            df[col] = parse_dates(df[col])

    print("Cleaning complete.")
    return df


def save_data(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'Cleaned_data.xlsx'
    df.to_excel(output_path)
    print(f'Cleaned data saved to {output_path}')
    return


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(df):

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found!")

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Convert target to numeric (extract numbers from text if needed)
    y = pd.to_numeric(y.apply(extract_numeric_from_text), errors='coerce')
    
    # Remove rows with invalid target values (NaN, infinity)
    valid_idx = y.notna() & np.isfinite(y)
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"Removed {(~valid_idx).sum()} rows with invalid target values. Remaining: {len(y)} rows")

    # Handle datetime columns (convert to ordinal)
    for col in X.select_dtypes(include=['datetime64[ns]']).columns:
        X[col] = X[col].map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)

    # Encode categorical columns
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))

    return X, y


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(X_train, y_train, task="regression"):

    if task == "regression":
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=RANDOM_STATE
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=RANDOM_STATE
        )

    model.fit(X_train, y_train)
    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test, task="regression"):

    predictions = model.predict(X_test)

    if task == "regression":
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
    else:
        acc = accuracy_score(y_test, predictions)
        print(f"Accuracy: {acc:.4f}")
    predictions = pd.DataFrame(predictions)

    return predictions


# ============================================================
# PIPELINE FUNCTIONS
# ============================================================

def run_pipeline(file_path, output_dir):

    # 1️⃣ Clean data
    df = clean_data(file_path)
    save_data(df, output_dir)

    # Auto detect task
    if df[TARGET_COLUMN].dtype in [int, float, np.int64, np.float64]:
        task = "regression"
    else:
        task = "classification"

    # 2️⃣ Prepare features
    X, y = prepare_features(df)

    # 3️⃣ Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # 4️⃣ Train model
    model = train_model(X_train, y_train, task)

    # 5️⃣ Evaluate
    predictions = model.predict(X_test)
   
    try:
        predictions_df = pd.DataFrame({'Predictions': predictions, 'Actual': y_test})
    except Exception as e:
        print(f"Prediction Error: {e}")
    
    predictions_df['Predictions'] = predictions_df['Predictions'].round(-2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_excel(output_dir / 'predictions_2.xlsx')

    # 6️⃣ Save model
    model.save_model(str(MODEL_PATH))
    print(f"Model saved as {MODEL_PATH}")

    return model, predictions


# ============================================================
# ANALYSIS UTILITY FUNCTIONS
# ============================================================

def load_data(file_path):
    file_path = Path(file_path)
    df = pd.read_excel(file_path)
    # Convert datetime columns to strings for Streamlit compatibility
    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[col] = df[col].astype(str)
    # Remove index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df


def basic_analysis(df):
    results = {}

    # Shape
    results["shape"] = df.shape

    # Summary statistics (only numeric columns to avoid datetime issues)
    results["summary"] = df.select_dtypes(include=[np.number]).describe()

    return results


def plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=(4, 3))
    if 'Age' in df.columns:
        ax.hist(df['Age'].dropna(), bins=15)
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
    else:
        ax.text(0.5, 0.5, 'Age column not available', ha='center', va='center')
        ax.set_title("Age Distribution")
    fig.tight_layout()
    return fig


def plot_missing_values(df):
    missing = df.isna().sum()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(missing.index, missing.values)
    ax.set_title("Missing Values")
    ax.tick_params(axis='x', rotation=90, labelsize=6)
    fig.tight_layout()
    return fig


def plot_ferritin_vs_tsat(df):
    fig, ax = plt.subplots(figsize=(4, 3))
    if 'Last Ferr' in df.columns and 'Last TSat' in df.columns:
        ax.scatter(df['Last Ferr'], df['Last TSat'])
        ax.set_title("Ferritin vs TSat")
        ax.set_xlabel("Last Ferr")
        ax.set_ylabel("Last TSat")
    else:
        ax.text(0.5, 0.5, 'Required columns not available', ha='center', va='center')
        ax.set_title("Ferritin vs TSat")
    fig.tight_layout()
    return fig


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

def main():
    # Load trained XGBoost model
    model = XGBRegressor()
    
    try:
        model.load_model(str(MODEL_PATH))
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        st.stop()

    # ----------------------------------------
    # Page Config
    # ----------------------------------------
    st.set_page_config(page_title="ESA Dose System", layout="wide")
    
    # ----------------------------------------
    # App Header
    # ----------------------------------------
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #1f77b4; font-size: 48px; margin-bottom: 5px;'>💊 Concepcion Calculator</h1>
        <p style='color: #666; font-size: 16px; margin-top: 0;'>ESA Dose Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------
    # Sidebar Navigation
    # ----------------------------------------
    page = st.sidebar.radio(
        "Navigation",
        ["Data Analysis", "ESA Dose Prediction"]
    )

    # ----------------------------------------
    # DATA ANALYSIS PAGE
    # ----------------------------------------
    if page == "Data Analysis":

        st.title("Dataset Analysis")

        try:
            df = load_data(DATA_FILE)
            results = basic_analysis(df)

            st.subheader("Dataset Shape")
            st.write(results["shape"])
            
            st.subheader("Summary Statistics")
            st.dataframe(results["summary"])

            st.subheader("Age Distribution")
            st.pyplot(plot_age_distribution(df), use_container_width=True)

            st.subheader("Missing Values in DataSet")
            st.pyplot(plot_missing_values(df), use_container_width=True)

            st.subheader("Ferritin vs TSat")
            st.pyplot(plot_ferritin_vs_tsat(df), use_container_width=True)

            if "correlation" in results:
                st.subheader("Correlation with New ESA Dose")
                st.dataframe(results["correlation"])

            st.divider()

            if st.button("Go To Prediction Page"):
                st.session_state.page = "ESA Dose Prediction"
                st.rerun()

        except Exception as e:
            st.error(f"Error loading file: {e}")

    # ----------------------------------------
    # PREDICTION PAGE
    # ----------------------------------------
    elif page == "ESA Dose Prediction":

        st.title("ESA Dose Prediction")
        left, center, right = st.columns([1, 2, 1])
        
        with left:
            with st.form("prediction_form"):

                age = st.number_input("Age", 0, 120, 50)
                current_dose = st.number_input("Current Dose", 0.0)
                last_ferr = st.number_input("Last Ferr", 0.0)
                last_tsat = st.number_input("Last TSat", 0.0)
                venofer_dose = st.number_input("Venofer Dose", 0.0)

                month = st.number_input("Month", 1, 12, 1)
                year = st.number_input("Year", 2000, 2100, 2025)

                jan = st.number_input("Start of Jan", 0.0)
                feb = st.number_input("Start of Feb", 0.0)
                mar = st.number_input("Start of Mar", 0.0)
                april = st.number_input("Start of April", 0.0)
                may = st.number_input("Start of May", 0.0)
                june = st.number_input("Start of June", 0.0)
                july = st.number_input("Start of July", 0.0)
                aug = st.number_input("Start of Aug", 0.0)
                sep = st.number_input("Start of Sep", 0.0)
                octo = st.number_input("Start of Oct", 0.0)
                nov = st.number_input("Start of Nov", 0.0)
                dec = st.number_input("Start of Dec", 0.0)
                jan1 = st.number_input("Start of Jan.1", 0.0)

                submit = st.form_submit_button("Predict ESA Dose")

        if submit:

            input_df = pd.DataFrame([{
                "Age": age,
                "Start of Jan": jan,
                "Start of Feb": feb,
                "Start of Mar": mar,
                "Start of April": april,
                "Start of May": may,
                "Start of June": june,
                "Start of July": july,
                "Start of Aug": aug,
                "Start of Sep": sep,
                "Start of Oct": octo,
                "Start of Nov": nov,
                "Start of Dec": dec,
                "Start of Jan.1": jan1,
                "Current Dose": current_dose,
                "Last Ferr": last_ferr,
                "Last TSat": last_tsat,
                "Venofer Dose": venofer_dose,
                "Month": month,
                "year": year,
                "Month.1": month,
                "year.1": year
            }])

            try:
                prediction = model.predict(input_df)[0] - current_dose 
                if prediction > 0:
                    st.success(f"Increase current dose by {round(float(prediction), -2)}")
                else:
                    st.success(f"Decrease current dose by {-round(float(prediction), -2)}")

            except Exception as e:
                st.error(f"Prediction Error: {e}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Check if model exists, if not run the pipeline
    if not MODEL_PATH.exists():
        print("Model not found. Running pipeline to train model...")
        output_dir = SCRIPT_DIR / "output"
        run_pipeline(str(DATA_FILE), str(output_dir))
        print("Pipeline completed! Starting Streamlit app...\n")
    
    main()
