import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def train_model(data_path='data/predictive_maintenance.csv'):
    """
    Trains a predictive maintenance model.
    Handles data loading, preprocessing, class imbalance (SMOTE), and evaluates performance.
    """
    # Load the dataset.
    df = pd.read_csv(data_path)

    # Separate features (X) from the target (y).
    # Drop identifiers and the target-related 'Failure Type'.
    X = df.drop(['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)
    y = df['Target']

    # Define categorical and numerical columns for preprocessing.
    categorical_features = ['Type']
    numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    # Create a preprocessor for one-hot encoding categorical features.
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])

    # Build the ML pipeline: preprocess, apply SMOTE for imbalance, then classify.
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('smote', SMOTE(random_state=42)),
                            ('classifier', RandomForestClassifier(random_state=42))])

    # Split data into training and testing sets, maintaining class proportions.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model.
    model.fit(X_train, y_train)

    # Make predictions on test data.
    y_pred = model.predict(X_test)

    # Evaluate the model.
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    return model

if __name__ == "__main__":
    # This block runs when the script is executed directly.
    import joblib
    print("Training the predictive maintenance model...")
    trained_model = train_model()
    model_save_path = 'models/predictive_maintenance_model.joblib'
    # Save the trained model for later use.
    joblib.dump(trained_model, model_save_path)
    print(f"Model training complete. Model saved to {model_save_path}")
