import pandas as pd
import joblib

def predict(data_path='data/predictive_maintenance.csv', model_path='models/predictive_maintenance_model.joblib'):
    """
    Loads a trained predictive maintenance model and uses it to make predictions on new data.
    """
    # Load the pre-trained model.
    model = joblib.load(model_path)

    # Load the data for prediction.
    df = pd.read_csv(data_path)

    # Prepare features (X) by dropping identifiers and target-related columns.
    X = df.drop(['UDI', 'Product ID', 'Target', 'Failure Type'], axis=1)

    # Make predictions using the loaded model.
    predictions = model.predict(X)

    # Add predictions to the DataFrame.
    df['Predicted_Target'] = predictions

    return df

if __name__ == "__main__":
    # This block runs when the script is executed directly.
    print("Making predictions using the trained model...")
    # Predict on the training data for demonstration. Replace with new data in a real application.
    predicted_df = predict()
    print("Predictions complete. Displaying first 5 rows with predictions:")
    print(predicted_df.head())
