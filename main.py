import argparse
import os

# Ensure script runs from project root for consistent pathing.
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# Import core functions for training and prediction.
from src.model_training import train_model
from src.predict import predict

def main():
    # Set up command-line argument parser.
    parser = argparse.ArgumentParser(description="Predictive Maintenance Project CLI")
    
    # Define command argument: 'train' or 'predict'.
    parser.add_argument('command', choices=['train', 'predict'], help='Action to perform: train model or make predictions.')
    
    # Define optional data path argument.
    parser.add_argument('--data_path', type=str, default='data/predictive_maintenance.csv', help='Path to the dataset.')
    
    # Define optional model path argument.
    parser.add_argument('--model_path', type=str, default='models/predictive_maintenance_model.joblib', help='Path to the trained model file.')

    # Parse arguments.
    args = parser.parse_args()

    # Execute command based on user input.
    if args.command == 'train':
        print("Executing: Model Training")
        train_model(data_path=args.data_path)
    elif args.command == 'predict':
        print("Executing: Prediction")
        predicted_df = predict(data_path=args.data_path, model_path=args.model_path)
        print("Predictions generated. Displaying first 5 rows:")
        print(predicted_df.head())

if __name__ == "__main__":
    # Run main function when script is executed directly.
    main()
