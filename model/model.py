import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(filepath):
    """Load the dataset safely."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at {filepath}")
    return pd.read_csv(filepath)

def preprocess_data(df):
    X = df['Text'] 
    y = df['Label'].map({'spam': 1, 'ham': 0}) if df['Label'].dtype == 'O' else df['Label']
    
    return X, y

def build_model_pipeline():
    """
    Creates a pipeline that converts text to numbers (TF-IDF) 
    and then trains the Random Forest.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    return pipeline

def main():
    # Setup paths (Adjust these to your local folders)
    data_path = 'data/emails.csv' 
    model_dir = 'models'
     # Add this to your main() function
    df = load_data(data_path)
    print("Available columns:", df.columns.tolist()) # This will reveal the true names
    # 1. Load
    try:
        df = load_data(data_path)
    except Exception as e:
        print(f"Error: {e}. Please ensure emails.csv is in the data folder.")
        return

    # 2. Preprocess
    X, y = preprocess_data(df)

    # 3. Split (Split BEFORE any processing to avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train
    print("Training model pipeline (Text Vectorization + Random Forest)...")
    model_pipeline = build_model_pipeline()
    model_pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model_pipeline.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    # Note: Ensure target_names match your specific dataset encoding
    print(classification_report(y_test, y_pred))

    # 6. Save
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model_pipeline, os.path.join(model_dir, 'spam_detector_model.pkl'))
    print(f"\nModel saved to {model_dir}")

if __name__ == "__main__":
    main()