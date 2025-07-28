# Predictive Maintenance for Energy Equipment

## Project Description

This project aims to develop a machine learning model that predicts equipment failure in a simulated energy sector environment. Using a public dataset of equipment sensor readings (e.g., temperature, pressure, vibration), we clean the data, engineer relevant features, and train a classification model to predict the likelihood of a component failing within a specific timeframe.

## Project Goals

*   To build a reliable classification model that can predict equipment failure.
*   To gain experience in data cleaning, feature engineering, and model evaluation.
*   To create a project that demonstrates skills relevant to the energy sector.

## Dataset

We use a public dataset of equipment sensor readings (`data/predictive_maintenance.csv`). The dataset contains sensor readings and labels indicating equipment failure.

### Data Acquisition (Optional: If downloading from Kaggle)

For convenience, the `predictive_maintenance.csv` dataset is already included in the `data/` directory of this repository. However, if you need to download it directly from Kaggle (e.g., for a fresh start or to verify the source), you can follow these steps:

1.  **Install the Kaggle API client:**
    ```bash
    pip install kaggle
    ```
2.  **Set up your Kaggle API credentials:**
    *   Go to your Kaggle account page (`https://www.kaggle.com/<your-username>/account`).
    *   Under the "API" section, click "Create New API Token". This will download a `kaggle.json` file.
    *   Move this `kaggle.json` file to `~/.kaggle/` (on Linux/macOS) or `C:\Users\<Windows-username>\.kaggle\` (on Windows).
    *   Ensure the permissions on `kaggle.json` are set to read/write only for your user (e.g., `chmod 600 ~/.kaggle/kaggle.json` on Linux/macOS).
3.  **Download the dataset:**
    *   The dataset can be found at: `https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification`
    *   Use the Kaggle API command to download the dataset to your `data/` directory:
    ```bash
    kaggle datasets download -d shivamb/machine-predictive-maintenance-classification -p data/
    ```
    *   Unzip the downloaded file if it's a `.zip` archive.

## Project Workflow

1.  **Data Acquisition:** The dataset is provided in `data/predictive_maintenance.csv`.

2.  **Data Cleaning and Preprocessing:** Handled within the `model_training.py` script, including one-hot encoding for categorical features.
3.  **Exploratory Data Analysis (EDA):** Performed in `notebooks/eda.ipynb` to understand data relationships and identify issues like class imbalance.
4.  **Feature Engineering:** Implicitly handled by selecting relevant numerical and categorical features.
5.  **Model Training:** A RandomForestClassifier is trained, with SMOTE applied to address class imbalance.
6.  **Model Evaluation:** The model's performance is evaluated using accuracy, precision, recall, and F1-score.
7.  **Model Persistence:** The trained model is saved for future use.
8.  **Prediction/Inference:** A dedicated script is provided to load the model and make predictions on new data.

## Project Structure

```
predictive_maintenance_energy_equipment/
├── data/
│   └── predictive_maintenance.csv  # Raw dataset
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis notebook
├── src/
│   ├── model_training.py           # Script for data preprocessing, model training, and evaluation
│   └── predict.py                  # Script for loading the trained model and making predictions
├── models/
│   └── predictive_maintenance_model.joblib # Saved trained model
├── venv/                           # Python virtual environment
├── main.py                         # Centralized command-line interface for the project
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd predictive_maintenance_energy_equipment
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The `main.py` script provides a centralized command-line interface for interacting with the project.

### Train the Model

To train the predictive maintenance model:

```bash
python main.py train
```

This will execute the `train_model` function from `src/model_training.py`, preprocess the data, train the RandomForestClassifier with SMOTE, evaluate its performance, and save the trained model to `models/predictive_maintenance_model.joblib`.

### Make Predictions

To make predictions using the trained model:

```bash
python main.py predict
```

This will load the `predictive_maintenance_model.joblib` and use it to make predictions on the `data/predictive_maintenance.csv` dataset (for demonstration purposes). In a real-world scenario, you would replace `data/predictive_maintenance.csv` with your new, unseen data.

### Custom Paths

You can specify custom paths for data and models:

```bash
python main.py train --data_path path/to/your/custom_data.csv
python main.py predict --data_path path/to/new_unseen_data.csv --model_path path/to/your/custom_model.joblib
```

## Results

Initial model training without addressing class imbalance showed a high overall accuracy but poor recall for the minority class (equipment failures). After incorporating SMOTE (Synthetic Minority Over-sampling Technique) into the training pipeline, the model's performance on the minority class significantly improved.

**Example Classification Report (after SMOTE):**

```
              precision    recall  f1-score   support

           0       0.99      0.98      0.98      1932
           1       0.52      0.68      0.59        68

    accuracy                           0.97      2000
   macro avg       0.75      0.83      0.78      2000
weighted avg       0.97      0.97      0.97      2000
```

*   **Accuracy:** Approximately 97% (slight decrease from initial 98.20% due to balancing classes).
*   **Recall (Failure Class - 1):** Improved from 0.53 to 0.68, indicating the model is now better at identifying actual failures.
*   **Precision (Failure Class - 1):** 0.52, meaning that when the model predicts a failure, it is correct about 52% of the time.

This balanced performance is crucial for predictive maintenance, where identifying as many true failures as possible (high recall) is often prioritized, even if it means a few more false alarms.

## Technologies Used

*   **Python:** The primary programming language for this project.
*   **Pandas:** For data manipulation and analysis.
*   **Scikit-learn:** For building and evaluating machine learning models.
*   **Imbalanced-learn (imblearn):** For handling class imbalance (SMOTE).
*   **Matplotlib/Seaborn:** For data visualization.
*   **Jupyter Notebook:** For interactive development and documentation.
*   **Joblib:** For model persistence.