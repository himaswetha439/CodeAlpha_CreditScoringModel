# Credit Scoring Model

## 1. Project Overview

This project develops a machine-learning classification system using the German Credit dataset from the UCI Machine Learning Repository.

The goal is to classify credit applicants into good-risk and bad-risk categories based on their financial and personal attributes.

Three machine-learning algorithms are compared:

* Logistic Regression
* Decision Tree
* Random Forest

## 2. Objectives

* Load and explore credit-related data.
* Preprocess numerical and categorical features.
* Train multiple classification models.
* Evaluate models using standard classification metrics.
* Compare the results and save a selected model.
* Visualize model performance.

## 3. Dataset

**Dataset:** Statlog (German Credit Data)

**Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

The dataset contains 1,000 applicants and 20 input features, with one target column.

The original target labels are:

* 1: Good credit risk
* 2: Bad credit risk

For this project, the labels are encoded as:

* 0: Good credit risk
* 1: Bad credit risk

## 4. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

## 5. Methodology

### Step 1: Data Loading

Load the dataset and validate its structure.

### Step 2: Data Preprocessing

Use one-hot encoding for categorical features and standardization for numerical features.

### Step 3: Train-Test Split

Split the dataset into 80% training data and 20% testing data, maintaining the class proportions.

### Step 4: Model Training

Train Logistic Regression, Decision Tree, and Random Forest classifiers.

### Step 5: Model Evaluation

Calculate:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Bad credit risk is treated as the positive class for the binary classification metrics.

### Step 6: Model Comparison

Compare the models using their evaluation metrics.

### Step 7: Model Saving

Select the model with the highest F1-score on the test evaluation and save it using Joblib.

## 6. Project Structure

```text
1_Credit_Scoring/
├── data/
│   └── german.data
├── models/
│   └── best_credit_scoring_model.joblib
├── outputs/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   └── confusion_matrix.png
├── src/
│   └── main.py
├── README.md
└── requirements.txt
```

The model and output files are generated when the program runs.

## 7. How to Run

1. Install Python.

2. Create and activate a virtual environment.

3. Install the dependencies:

   `python -m pip install -r 1_Credit_Scoring/requirements.txt`

4. Place `german.data` inside the project's `data` folder.

5. Run the program from the internship's main folder:

   `python 1_Credit_Scoring/src/main.py`

## 8. Expected Outputs

After a successful run, the program generates:

* A CSV file containing model evaluation metrics.
* A graph comparing the three classifiers.
* A confusion matrix for the selected model.
* A saved machine-learning pipeline.

Actual evaluation scores depend on the data split and model behavior. They should be reported only after running the program.

## 9. Limitations and Responsible Use

This project is an educational demonstration, not a validated credit-underwriting system.

Historical credit data may contain biases and may not represent current applicants or lending conditions. Model scores do not establish that a model is fair, reliable, or suitable for real-world lending decisions.

The model must not be used to make actual lending decisions without appropriate validation, fairness assessments, regulatory review, and human oversight.

## 10. Conclusion

This project demonstrates a complete introductory machine-learning workflow, including data preprocessing, classification, evaluation, model comparison, visualization, and model persistence.
