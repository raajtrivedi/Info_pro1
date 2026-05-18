import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_and_evaluate():
    print("Loading data from data/cleaned_data.csv...")
    try:
        df = pd.read_csv("data/cleaned_data.csv")
    except FileNotFoundError:
        print("Error: data/cleaned_data.csv not found.")
        return

    # Prepare data based on the existing pipeline logic
    top10 = df['Complaint Type'].value_counts().head(10).index
    df = df[df['Complaint Type'].isin(top10)]
    # Use clean_text if it exists, otherwise fallback to the raw text (assuming it's called 'complaint')
    text_col = 'clean_text' if 'clean_text' in df.columns else df.columns[0]
    
    # Fill NaNs just in case
    df[text_col] = df[text_col].fillna('')
    df['combined_text'] = df[text_col]  # Fixed Target Leakage

    X = df['combined_text']
    y = df['Complaint Type']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
        ('model', LinearSVC(class_weight='balanced', max_iter=2000, random_state=42, dual="auto"))
    ])

    print("Training the ML pipeline (TF-IDF + LinearSVC)...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model on test data...")
    y_pred = pipeline.predict(X_test)

    # Output Metrics
    print("\n" + "="*50)
    print("MODEL METRICS & EVALUATION")
    print("="*50)
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("="*50)

    # Save the newly trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/complaint_model.pkl")
    print("Model successfully saved to models/complaint_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()
