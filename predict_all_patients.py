# ============================================================
# BATCH PREDICTION SCRIPT FOR ALL PATIENTS
# ============================================================
import pandas as pd
import numpy as np
import re
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "predictions_output"
MODEL_PATH = SCRIPT_DIR / "xgboost_model.pkl"
CLEANED_DATA_PATH = SCRIPT_DIR / "output" / "Cleaned_data.xlsx"
DATA_FILE = DATA_DIR / "Salman Bhai project data.xlsx"

TARGET_COLUMN = "New ESA Dose"

COLUMNS_TO_DROP = [
    'Unnamed: 31',
    'Unnamed: 2',
    'Schedule',
    'Effective Date',
    'Dose Change',
    'New Dose Effective Date',
    'HGB redraw',
    'Mid Month HGB',
    'Mid Month Dose Change',
    'New ESA Dose2',
    'New Dose2 Effective Date',
    'Comments',
    'Effective Date.1',
    'Venofer Dose/ Additional Comments'  # Will be replaced with 'Venofer Dose'
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def detect_dominant_type(series):
    """Detect the dominant data type in a series"""
    from collections import Counter
    types = series.dropna().apply(lambda x: type(x)).tolist()
    if not types:
        return None
    return Counter(types).most_common(1)[0][0]


def coerce_to_dominant_type(series):
    """Coerce series to its dominant type"""
    dominant_type = detect_dominant_type(series)

    if dominant_type in [int, float, np.int64, np.float64]:
        return pd.to_numeric(series, errors="coerce")

    if dominant_type == str:
        return series.astype(str)

    return series


def parse_dates(series):
    """Parse date columns"""
    return pd.to_datetime(series, errors="coerce")


def extract_numeric_from_text(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    match = re.search(r'(\d+\.?\d*)', str(value))
    if match:
        return float(match.group(1))
    return np.nan


def clean_data(df):
    """Apply data cleaning transformations (matching Concepcion_app.py logic)"""
    
    # Create patient identifier from Unnamed: 0 or use index
    if 'Unnamed: 0' in df.columns and df['Unnamed: 0'].notna().any():
        # Keep Unnamed: 0 as is
        pass
    else:
        # Create numeric index if Unnamed: 0 is all NaN
        df['Unnamed: 0'] = range(len(df))
    
    # Rename Unnamed: 2 to Age (this is the age column in original data)
    if 'Unnamed: 2' in df.columns:
        df = df.rename(columns={'Unnamed: 2': 'Age'})
    
    # Rename long column names to short
    if 'Venofer Dose/ Additional Comments' in df.columns:
        df = df.rename(columns={'Venofer Dose/ Additional Comments': 'Venofer Dose'})
    
    # Parse and extract date columns BEFORE dropping them
    # This needs to happen before COLUMNS_TO_DROP is applied
    if 'Effective Date' in df.columns:
        df['Effective Date'] = parse_dates(df['Effective Date'])
        df['Month'] = df['Effective Date'].dt.month
        df['year'] = df['Effective Date'].dt.year
    
    if 'Effective Date.1' in df.columns:
        df['Effective Date.1'] = parse_dates(df['Effective Date.1'])
        df['Month.1'] = df['Effective Date.1'].dt.month
        df['year.1'] = df['Effective Date.1'].dt.year
    
    # Drop unnecessary columns (keeping 'Unnamed: 0' as index and 'Age')
    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns],
                 errors="ignore")
    
    # Coerce dominant types
    for col in df.columns:
        df[col] = coerce_to_dominant_type(df[col])
    
    # Extract dose values automatically
    for col in df.columns:
        if "dose" in col.lower():
            df[col] = df[col].apply(extract_numeric_from_text)
    
    return df


def prepare_features_for_prediction(df):
    """Prepare features for prediction (handles missing target column, preserves index)"""
    X = df.copy()
    
    # IMPORTANT: Keep 'Unnamed: 0' column as patient identifier
    patient_index = None
    if 'Unnamed: 0' in X.columns:
        patient_index = X['Unnamed: 0'].copy()
        X = X.drop(columns=['Unnamed: 0'])
    
    # Handle datetime columns (convert to ordinal)
    for col in X.select_dtypes(include=['datetime64[ns]']).columns:
        X[col] = X[col].map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
    
    # Encode categorical columns
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))
    
    return X, patient_index


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_all_patients():
    """Load model and make predictions for all patients"""
    
    print("="*60)
    print("BATCH PREDICTION FOR ALL PATIENTS")
    print("="*60)
    
    # Step 1: Load the trained model
    print("\n[1/6] Loading trained model...")
    try:
        model = XGBRegressor()
        model.load_model(str(MODEL_PATH))
        print(f"✓ Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"✗ Model file not found at {MODEL_PATH}")
        print("  Please train the model first using Concepcion_app.py")
        return None
    
    # Step 2: Load the original data
    print("\n[2/6] Loading original dataset...")
    try:
        df = pd.read_excel(DATA_FILE)
        print(f"✓ Loaded data from {DATA_FILE}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None
    
    original_df = df.copy()
    print(f"  Total patients: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    
    # Step 3: Clean data (apply same transformations as Concepcion_app.py)
    print("\n[3/6] Cleaning and preprocessing data...")
    try:
        df = clean_data(df)
        print(f"✓ Data cleaning complete")
    except Exception as e:
        print(f"✗ Error during data cleaning: {e}")
        return None
    
    # Step 4: Remove target column if present (for prediction)
    print("\n[4/6] Preparing features for prediction...")
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])
        print(f"✓ Removed target column '{TARGET_COLUMN}'")
    
    # Step 5: Apply feature engineering while preserving patient index
    print("\n[5/6] Applying feature engineering...")
    try:
        X, patient_index = prepare_features_for_prediction(df)
        print(f"✓ Feature preparation complete")
        print(f"  Feature shape: {X.shape}")
        print(f"  Samples with patient ID: {len(patient_index)}")
    except Exception as e:
        print(f"✗ Error in feature preparation: {e}")
        return None
    
    # Step 6: Make predictions
    print("\n[6/6] Making predictions...")
    try:
        predictions = model.predict(X)
        print(f"✓ Predictions complete")
        print(f"  Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        return None
    
    # ============================================================
    # CREATE RESULTS DATAFRAME
    # ============================================================
    
    print("\n" + "="*60)
    print("CREATING RESULTS DATAFRAME")
    print("="*60)
    
    results_df = pd.DataFrame()
    
    # Add patient identifier as first column
    if patient_index is not None:
        results_df['Patient_ID'] = patient_index.values
        print(f"✓ Added patient identifiers")
    
    # Add original ESA dose if available
    if TARGET_COLUMN in original_df.columns:
        original_dose_values = pd.to_numeric(original_df[TARGET_COLUMN].apply(extract_numeric_from_text), errors='coerce')
        results_df['Original_ESA_Dose'] = original_dose_values.values
    
    # Add predictions
    results_df['Predicted_ESA_Dose'] = predictions.round(2)
    
    # Calculate dose adjustment
    if TARGET_COLUMN in original_df.columns:
        original_dose_values = pd.to_numeric(original_df[TARGET_COLUMN].apply(extract_numeric_from_text), errors='coerce')
        results_df['Dose_Adjustment'] = (predictions - original_dose_values.values).round(2)
        results_df['Adjustment_Type'] = results_df['Dose_Adjustment'].apply(
            lambda x: "Increase" if x > 0 else ("Decrease" if x < 0 else "No Change")
        )
    
    # Add other relevant columns (Age, Ferritin, etc.) from original data
    relevant_cols = ['Age', 'Last Ferr', 'Last TSat', 'Current Dose', 'Venofer Dose']
    for col in relevant_cols:
        if col in original_df.columns:
            results_df[col] = original_df[col].values
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to Excel
    output_excel = OUTPUT_DIR / f"predictions_{timestamp}.xlsx"
    results_df.to_excel(output_excel, index=False)
    print(f"✓ Predictions saved to: {output_excel}")
    
    # Save to CSV
    output_csv = OUTPUT_DIR / f"predictions_{timestamp}.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"✓ CSV saved to: {output_csv}")
    
    # ============================================================
    # PRINT SUMMARY STATISTICS
    # ============================================================
    
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    
    print(f"\nTotal Patients: {len(results_df)}")
    print(f"\nPredicted ESA Dose Statistics:")
    print(f"  Mean: {predictions.mean():.2f}")
    print(f"  Std Dev: {predictions.std():.2f}")
    print(f"  Min: {predictions.min():.2f}")
    print(f"  Max: {predictions.max():.2f}")
    
    if 'Dose_Adjustment' in results_df.columns:
        print(f"\nDose Adjustment Summary:")
        print(results_df['Adjustment_Type'].value_counts())
    
    print(f"\n✓ Process completed successfully!")
    print(f"\nFirst 10 predictions:")
    print(results_df.head(10).to_string(index=False))
    
    return results_df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    results = predict_all_patients()
