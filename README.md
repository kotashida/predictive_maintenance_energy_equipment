# Predictive Maintenance for Energy Equipment

## Project Description

This project develops a robust machine learning model to predict equipment failure in a simulated energy sector environment. Leveraging a dataset of equipment sensor readings, this analysis demonstrates a comprehensive workflow from data exploration to model deployment. The primary objective is to build a highly reliable classification model that can preemptively identify potential failures, thereby minimizing downtime and reducing operational costs.

## Key Quantitative Skills

*   **Statistical Analysis:** Performed exploratory data analysis (EDA) to identify trends, correlations, and anomalies in sensor data. Investigated class imbalance and its impact on model performance.
*   **Data Preprocessing:** Implemented one-hot encoding for categorical variables to prepare the data for machine learning algorithms.
*   **Feature Engineering:** Selected relevant features and engineered new ones to improve model accuracy.
*   **Machine Learning Modeling:**
    *   Trained a **RandomForestClassifier**, chosen for its high accuracy, robustness to overfitting, and ability to handle non-linear relationships between features.
    *   Addressed significant class imbalance using the **Synthetic Minority Over-sampling Technique (SMOTE)**, which synthetically generates new minority class instances to create a balanced dataset. This was a critical step to prevent the model from being biased towards the majority class (non-failure).
*   **Model Evaluation:**
    *   Evaluated the model using a comprehensive set of metrics: **accuracy, precision, recall, and F1-score**.
    *   Analyzed the **confusion matrix** to understand the trade-offs between Type I and Type II errors.
    *   Focused on **recall** as a key performance indicator, as maximizing the detection of true failures is paramount in a predictive maintenance context.
*   **Programming & Tooling:** Utilized Python with libraries such as Pandas, Scikit-learn, and Matplotlib to perform analysis and build the model.

## Methodology

1.  **Exploratory Data Analysis (EDA):** The initial analysis, conducted in `notebooks/eda.ipynb`, revealed a significant class imbalance in the dataset: only 3.4% of the instances represented equipment failures. This finding was critical, as a naive model would achieve high accuracy by simply predicting "no failure" for all instances, while failing to identify actual problems. The EDA also included a correlation analysis, which showed that `Torque` and `Rotational speed` were moderately correlated with equipment failure.

2.  **Data Preprocessing:** The `Type` feature, being categorical, was converted into a numerical format using one-hot encoding. This is a standard technique to prevent the model from assuming an ordinal relationship between categories.

3.  **Model Training Strategy:**
    *   **Classifier Selection:** A **RandomForestClassifier** was chosen due to its ability to capture complex interactions between features and its resilience to overfitting, which is often a concern with high-dimensional data.
    *   **Addressing Class Imbalance:** To counteract the class imbalance identified in the EDA, **SMOTE (Synthetic Minority Over-sampling Technique)** was integrated into the training pipeline. SMOTE creates synthetic samples of the minority class (failures), forcing the model to learn the characteristics of failures more effectively. This was a deliberate choice to improve the model's sensitivity to the minority class, which is crucial for a predictive maintenance application.

4.  **Model Evaluation:** The model was evaluated on a held-out test set (20% of the data). The performance was assessed using a classification report and a confusion matrix, with a particular focus on the trade-off between precision and recall for the "failure" class.

## Quantified Results

The model's performance was significantly improved by the application of SMOTE. The final evaluation on the test set yielded the following results:

**Classification Report (after SMOTE):**

```
              precision    recall  f1-score   support

           0       0.99      0.98      0.98      1932
           1       0.52      0.68      0.59        68

    accuracy                           0.97      2000
   macro avg       0.75      0.83      0.78      2000
weighted avg       0.97      0.97      0.97      2000
```

*   **Overall Accuracy:** The model achieved an accuracy of **97%**.
*   **Recall (Failure Class - 1):** The recall for the failure class is **0.68**. This is the most critical metric for this application, as it indicates that the model successfully identified **68% of all actual equipment failures**. This is a substantial improvement from a baseline model that would have a recall of 0 for the failure class.
*   **Precision (Failure Class - 1):** The precision for the failure class is **0.52**. This means that when the model predicts a failure, it is correct **52% of the time**. While this indicates a number of false positives, in a predictive maintenance scenario, a false positive (unnecessary inspection) is often preferable to a false negative (a missed failure).
*   **F1-Score:** The F1-score, which is the harmonic mean of precision and recall, is **0.59** for the failure class, indicating a reasonable balance between the two metrics.

These results demonstrate a well-balanced model that is effective at its primary goal: identifying a majority of equipment failures while maintaining a manageable rate of false alarms.

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
    git clone https://github.com/kotashida/predictive_maintenance_energy_equipment
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

## Technologies Used

*   **Python:** The primary programming language for this project.
*   **Pandas:** For data manipulation and analysis.
*   **Scikit-learn:** For building and evaluating machine learning models.
*   **Imbalanced-learn (imblearn):** For handling class imbalance (SMOTE).
*   **Matplotlib/Seaborn:** For data visualization.
*   **Jupyter Notebook:** For interactive development and documentation.
*   **Joblib:** For model persistence.
