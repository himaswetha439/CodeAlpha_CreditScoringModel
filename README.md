# Credit Scoring Model

## 1. Project Overview

This project develops a machine learning classification system using the German Credit dataset from the UCI Machine Learning Repository.

The objective is to classify credit applicants into two categories:

* **Good Credit Risk**
* **Bad Credit Risk**

Three machine learning algorithms are implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The project demonstrates a complete machine learning workflow, including data preprocessing, model training, cross-validation, evaluation, visualization, and model persistence.

## 2. Project Objectives

* Load and explore credit-related data.
* Preprocess numerical and categorical features.
* Train multiple classification models.
* Apply stratified cross-validation for model selection.
* Evaluate model performance using standard classification metrics.
* Compare the results of different algorithms.
* Save the selected model for future predictions.
* Generate visualizations to understand model performance.

## 3. Dataset

**Dataset:** Statlog (German Credit Data)

**Source:** UCI Machine Learning Repository

**Dataset link:** https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

The dataset contains:

* 1,000 applicant records.
* 20 input features.
* 1 target column representing credit risk.

### Original Target Labels

| Original Label | Meaning          |
| -------------- | ---------------- |
| 1              | Good credit risk |
| 2              | Bad credit risk  |

### Target Encoding

For binary classification, the target labels are encoded as follows:

| Encoded Label | Meaning          |
| ------------- | ---------------- |
| 0             | Good credit risk |
| 1             | Bad credit risk  |

Bad credit risk is treated as the positive class when calculating Precision, Recall, and F1-score.

## 4. Technologies Used

* **Programming Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Data Visualization:** Matplotlib
* **Model Persistence:** Joblib
* **Development Environment:** Visual Studio Code
* **Version Control:** Git and GitHub

## 5. Machine Learning Algorithms

### 5.1 Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of an applicant belonging to a particular credit-risk category.

### 5.2 Decision Tree

A Decision Tree classifies applicants by learning decision rules from their input features.

### 5.3 Random Forest

Random Forest combines predictions from multiple decision trees to produce a classification result.

All three algorithms are evaluated using the same training and test split to support a consistent comparison.

## 6. Methodology

### Step 1: Data Loading

Load the German Credit dataset from the project's `data` directory.

Validate the dataset structure, check for missing values, and verify the target labels.

### Step 2: Data Preprocessing

The dataset contains both numerical and categorical features.

* **StandardScaler:** Standardizes numerical features.
* **OneHotEncoder:** Converts categorical features into numerical representations.
* **ColumnTransformer:** Applies the appropriate preprocessing operations to each feature type.
* **Pipeline:** Combines preprocessing and classification into a single workflow.

The preprocessing steps are fitted within each training fold during cross-validation to avoid data leakage.

### Step 3: Train-Test Split

Split the dataset into:

* **80% training data**
* **20% testing data**

Stratified splitting maintains approximately the same class proportions in both sets.

The test set is kept separate from the cross-validation process used for model selection.

### Step 4: Model Training and Cross-Validation

The following models are evaluated:

* Logistic Regression
* Decision Tree
* Random Forest

Stratified 5-fold cross-validation is performed on the training data.

The training data is divided into five folds. Each fold is used for validation while the remaining folds are used for training, repeating the process five times.

The mean cross-validation F1-score for the bad-credit-risk class is used to select the model.

**Model selection is based on training-data cross-validation, not on test-set performance.**

### Step 5: Model Evaluation

The models are evaluated on the held-out test set using the following metrics:

* **Accuracy:** Proportion of all predictions that are correct.
* **Precision:** Proportion of predicted bad-risk applicants who actually belong to the bad-risk class.
* **Recall:** Proportion of actual bad-risk applicants correctly identified.
* **F1-score:** Harmonic mean of Precision and Recall.
* **ROC-AUC:** Measures the model's ability to distinguish between the two classes across classification thresholds.

These metrics provide different perspectives on model performance.

### Step 6: Model Comparison

The three algorithms are compared using their mean cross-validation F1-scores and test-set evaluation metrics.

The comparison results are saved as a CSV file and visualized using a bar chart.

### Step 7: Model Saving

The model with the highest mean cross-validation F1-score is selected and saved using Joblib.

The saved pipeline includes the preprocessing operations and the trained classifier, allowing it to process new applicant records.

### Step 8: Confusion Matrix

A confusion matrix is generated for the selected model using its predictions on the held-out test set.

The matrix displays the correctly and incorrectly classified applicants in the good-risk and bad-risk categories.

## 7. Model Evaluation Results

The following results were obtained during the current execution of the project.

| Model               | Mean CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
| ------------------- | ---------: | ------------: | -------------: | ----------: | ------: | -----------: |
| Logistic Regression |     0.5696 |        0.7500 |         0.5581 |      0.8000 |  0.6575 |       0.8058 |
| Decision Tree       |     0.4822 |        0.6600 |         0.4355 |      0.4500 |  0.4426 |       0.6000 |
| Random Forest       |     0.5722 |        0.7450 |         0.5763 |      0.5667 |  0.5714 |       0.7998 |

**Selected model: Random Forest**

Random Forest achieved the highest mean cross-validation F1-score of **0.5722** and was therefore selected according to the predefined model-selection procedure.

Logistic Regression achieved a higher F1-score and accuracy on this particular test split. However, the selected model was not changed based on test-set results.

These results represent one experimental split and should not be interpreted as a guarantee of performance on other datasets.

## 8. Project Structure

```text
1_Credit_Scoring/
│
├── data/
│   ├── german.data
│   ├── german.data-numeric
│   ├── german.doc
│   └── Index
│
├── models/
│   └── best_credit_scoring_model.joblib
│
├── outputs/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   └── confusion_matrix.png
│
├── src/
│   ├── main.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

The model and output files are generated when the training program runs.

## 9. Installation and Execution

### Prerequisites

* Python 3.10 or a compatible Python version.
* pip package manager.
* Git, if you want to clone the repository.

### Step 1: Clone the Repository

```bash
git clone https://github.com/himaswetha439/CodeAlpha_CreditScoringModel.git
```

### Step 2: Navigate to the Project Directory

```bash
cd CodeAlpha_CreditScoringModel
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Verify the Dataset

Ensure that `german.data` is present inside the `data` directory.

### Step 5: Train and Evaluate the Models

Run:

```bash
python src/main.py
```

The program loads the dataset, performs preprocessing, evaluates the models using cross-validation and test-set metrics, selects the best model based on mean cross-validation F1-score, and saves the resulting files.

### Step 6: Generate a Sample Prediction

After successfully training the model, run:

```bash
python src/predict.py
```

The prediction script loads the saved model and demonstrates a prediction for a sample applicant.

**Note:** Run the commands from inside the `1_Credit_Scoring` project directory.

## 10. Generated Outputs

The project generates the following files:

| File                               | Description                                                            |
| ---------------------------------- | ---------------------------------------------------------------------- |
| `model_comparison.csv`             | Cross-validation and test-set evaluation metrics for the three models. |
| `model_comparison.png`             | Bar chart comparing model evaluation metrics.                          |
| `confusion_matrix.png`             | Confusion matrix for the selected model.                               |
| `best_credit_scoring_model.joblib` | Saved preprocessing and classification pipeline.                       |

These outputs are stored in the `outputs` and `models` directories.

## 11. Limitations and Responsible Use

This project is an educational machine learning demonstration. It is not a validated credit-underwriting system.

Important limitations include:

* The dataset contains historical information and may not represent current applicants.
* Historical data may contain biases that affect model predictions.
* Evaluation results depend on the dataset, split, and model configuration.
* Classification metrics alone do not establish fairness or suitability for real-world lending.
* Predicted probabilities should not automatically be interpreted as calibrated real-world risk estimates.

The model should not be used to make actual lending decisions without appropriate validation, fairness assessments, regulatory review, and human oversight.

## 12. Future Improvements

Possible future improvements include:

* Hyperparameter tuning using cross-validation.
* Evaluation on additional datasets.
* Probability calibration.
* Fairness and bias assessment.
* Feature importance analysis.
* Improved handling of class imbalance.
* More comprehensive testing and documentation.

## 13. Conclusion

This project demonstrates an end-to-end machine learning workflow for credit-risk classification using the German Credit dataset.

It implements three classification algorithms, applies preprocessing through machine learning pipelines, uses stratified cross-validation for model selection, evaluates performance on a separate test set, visualizes evaluation metrics, and saves the selected model for future predictions.

The project provides practical experience with supervised learning, model evaluation, cross-validation, data preprocessing, and responsible machine learning development.
