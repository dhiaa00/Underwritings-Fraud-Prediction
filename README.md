# Underwritings-Fraud-Prediction

## What is Underwriting?
![img](https://fjwp.s3.amazonaws.com/blog/wp-content/uploads/2020/07/24133622/Underwriter.png)
Underwriting is the process of assessing and quantifying the financial risk associated with an individual or institution, typically involving loans, insurance, or investments. Financial institutions like banks, insurance agencies, and loan companies conduct underwriting. 

For further information, refer to: [What Is Underwriting?](https://www.investopedia.com/terms/u/underwriting.asp)

## Underwriter's Role

An underwriter evaluates the risk of insuring a home, car, or person. They determine:

* Whether to accept or reject an application for a loan.
* The terms and conditions of the policy.

## Rise of Machine Learning in Underwriting

Underwriting has gained significant importance in recent years, leading to increased investment in the process. 
Machine learning (ML) and Artificial Intelligence (AI) are being employed as solutions to assist underwriters in making better decisions. MoneyLion is one such company leveraging these technologies.

## Project Description

The MoneyLion team has allocated a significant budget to develop a high-accuracy model for fraud detection in their loan operations using machine learning and deep learning techniques.

The IT team has gathered a substantial amount of data and provided a sample for this project. Your objective is to build a model capable of predicting the likelihood of fraud occurring in loan applications.

**Fraud Score:**

The probability of fraud is represented as a score between 0 and 1000, where a higher score indicates a lower probability of fraud. Your task is to predict this score.

Refer to the "DATA" section for a detailed understanding of the provided dataset.

## Data

The `data.zip` file contains three folders:

* **train:** Contains a single JSON file holding variables used for loan underwriting from MoneyLion's data provider. Each row represents an underwriting report.
* **dictionaries:** Contains an Excel file with detailed descriptions of each underwriting variable.
* **submission:** Contains a CSV file for you to make predictions. Fill the `submission.csv` with your predicted scores based on the provided data and submit it to the platform for evaluation.

## Project: Predicting Fraud Score (Using Colaboratory Notebook)

This repository contains the Python code for a project that attempts to predict fraud scores using a machine learning approach.

**Original Notebook:** https://colab.research.google.com/ (**Note:** This link points to the original notebook in Google Colaboratory and might not be publicly accessible without required permissions.)

**Dependencies:**

* pandas (for data manipulation)
* numpy (for numerical operations)
* sklearn (for machine learning algorithms)
    * train_test_split (for splitting data into training and testing sets)
    * LabelEncoder (for encoding categorical variables)
    * OneHotEncoder (for creating dummy variables from categorical features)
* xgboost (for XGBoost algorithm)
* sklearn.ensemble (for Bagging and Random Forest Regressor models)
* sklearn.model_selection (for GridSearchCV)

**Data:**

The code assumes a CSV file named "train.csv" located in the "./data/train/" directory. This file should contain the features used for training the model.

**Code Description:**

1. **Data Loading and Cleaning:**
   * Loads the data from the JSON file using pandas.
   * Cleans the column names.
   * Drops irrelevant columns and features with high missing values.
   * Handles missing values.
   * Encodes categorical columns using LabelEncoder.
   * Creates dummy variables for relevant features using OneHotEncoder.

2. **Model Training:**

   * **XGBoost Model:**
     * Trains an XGBoost model with specified parameters.
     * Saves the trained model.
     * Evaluates the model's performance using R-squared and RMSE metrics.

   * **Ensemble Learning:**
     * Trains a BaggingRegressor model.
     * Evaluates the model's performance.

   * **Random Forest Regression:**
     * Trains a RandomForestRegressor model.
     * Evaluates the model's performance.

   * **GridSearchCV for Best Parameters:**
     * Defines a parameter grid for the RandomForestRegressor.
     * Performs GridSearchCV to find the best hyperparameters.
     * Trains the model with the best parameters.
     * Evaluates the model's performance.

**Note:**

* This script is a basic example and might require further tuning and feature engineering for optimal performance.
* Saving the trained model requires specifying a valid path in `model.save_model('./your_model_path.model')`.

## Getting Started

**Prerequisites:**

1. **Python 3.x:**
   - Verify installation by running `python --version` or `python3 --version` in your terminal/command prompt.
   - If not installed or incorrect version, download and install from the official website: https://www.python.org/downloads/

**Install Required Libraries:**

1. Open a terminal/command prompt and navigate to the project directory.
2. Run the following command to install libraries using pip:

   ```bash
   pip install pandas numpy scikit-learn xgboost
   ```
   This installs pandas, numpy, scikit-learn, and xgboost libraries.

**Data Preparation:**

  1. Extract the contents of the data.zip file.
  2. Maintain the extracted folder structure (containing "train", "dictionaries", and "submission" folders).
  3. Run the Script:

  Navigate to the directory containing the script (e.g., with the Python file).

  Run the script using the following command:

    ```Bash
    python your_script_name.py
    ```
  Replace your_script_name.py with the actual filename (e.g., main.py or fraud_prediction.py).

**Understanding the Output:**

  The script executes the code, performing:
  1. Data loading and cleaning.
  2. Training the models.
  3. Evaluating their performance.
  
  The output displays the R-squared and RMSE scores of each model for comparison.

**Additional Notes:**
  
  ### This script serves as a foundation. Consider further exploration:
    1. Feature engineering for extracting informative features.
    2. Hyperparameter tuning using GridSearchCV or similar techniques.
    3. Experimenting with different machine learning algorithms.
    4. Replace the placeholder ./your_model_path.model with the desired path to save the trained model for future predictions.
  Following these steps allows you to set up the environment, run the script, and gain insights into the model performance for fraud score prediction.
