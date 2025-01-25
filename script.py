from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import pandas as pd
import argparse

# Function to load the model for deployment
def model_fn(model_dir):
    """Load the trained model for deployment"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == "__main__":
    print("[INFO] Extracting arguments")

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # SageMaker-specific arguments: model, train, and test directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")

    args, _ = parser.parse_known_args()

    print("[INFO] Reading data")

    # Load training and testing datasets
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop(-1)  # Remove the target column from features

    print("[INFO] Building training and testing datasets")
    X_train = train_df[features]
    y_train = train_df[label]
    X_test = test_df[features]
    y_test = test_df[label]

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

    # Train the Random Forest model
    print("[INFO] Training the Random Forest model")
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=-1, verbose=2)
    model.fit(X_train, y_train)

    # Save the model
    model_path = os.path.join(args.model-dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved to {model_path}")

    # Evaluate the model
    print("[INFO] Evaluating the model")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Test Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
