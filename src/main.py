from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "german.data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD THE DATASET
# ============================================================

column_names = [
    "checking_account",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    "credit_risk",
]


def load_dataset():
    """Load the German Credit dataset and validate its structure."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Place german.data inside the project's data folder."
        )

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=column_names,
    )

    if df.shape != (1000, 21):
        raise ValueError(
            f"Unexpected dataset shape: {df.shape}. "
            "Expected 1000 rows and 21 columns."
        )

    if df.isnull().any().any():
        raise ValueError(
            "The dataset contains missing values. "
            "Inspect the data before training."
        )

    if not df["credit_risk"].isin([1, 2]).all():
        raise ValueError("Unexpected values found in the target column.")

    print("=" * 60)
    print("CREDIT SCORING MODEL")
    print("=" * 60)

    print("\nDataset loaded successfully!")
    print(f"Number of applicants: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nFirst five records:")
    print(df.head())

    print("\nOriginal credit risk distribution:")
    print(df["credit_risk"].value_counts().sort_index())

    return df


# ============================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data(df):
    """Separate input features from the prediction target."""

    X = df.drop(columns=["credit_risk"])

    # Original label 1 = good credit risk
    # Original label 2 = bad credit risk
    # We encode bad credit risk as 1 so that it is the positive class.
    y = df["credit_risk"].map({
        1: 0,
        2: 1,
    })

    print("\nFeatures (X) shape:", X.shape)
    print("Target (y) shape:", y.shape)

    print("\nTarget meaning:")
    print("0 = Good credit risk")
    print("1 = Bad credit risk")

    print("\nTarget distribution after encoding:")
    print(y.value_counts().sort_index())

    return X, y


# ============================================================
# 4. CREATE THE PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor(X):
    """Encode categorical features and scale numerical features."""

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_columns,
            ),
        ]
    )

    print("\nNumerical features:", len(numerical_columns))
    print("Categorical features:", len(categorical_columns))

    return preprocessor


# ============================================================
# 5. CREATE THE MACHINE LEARNING MODELS
# ============================================================

def create_models(X):
    """Create three classification pipelines."""

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", create_preprocessor(X)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]),

        "Decision Tree": Pipeline([
            ("preprocessor", create_preprocessor(X)),
            (
                "classifier",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", create_preprocessor(X)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]),
    }

    return models


# ============================================================
# 6. TRAIN AND EVALUATE THE MODELS
# ============================================================

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Train each model and calculate evaluation metrics."""

    results = []
    trained_models = {}

    for model_name, model in models.items():

        print("\n" + "-" * 60)
        print(f"Training: {model_name}")
        print("-" * 60)

        # Train the model using training data.
        model.fit(X_train, y_train)

        # Predict credit risk for unseen test data.
        predictions = model.predict(X_test)

        # Probability of the positive class: bad credit risk.
        probabilities = model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics.
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
        })

        trained_models[model_name] = model

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

    results_df = pd.DataFrame(results)

    return results_df, trained_models


# ============================================================
# 7. SAVE RESULTS AND COMPARE MODELS
# ============================================================

def save_results(results_df):
    """Save the evaluation results and create a comparison graph."""

    results_path = OUTPUT_DIR / "model_comparison.csv"

    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(results_df.to_string(index=False))

    # Plot the model comparison.
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
    ]

    ax = results_df.set_index("Model")[metrics].plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title("Credit Scoring Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()

    graph_path = OUTPUT_DIR / "model_comparison.png"

    plt.savefig(graph_path, dpi=300)
    plt.close()

    print(f"\nResults saved to: {results_path}")
    print(f"Comparison graph saved to: {graph_path}")


# ============================================================
# 8. SAVE THE BEST MODEL
# ============================================================

def save_best_model(results_df, trained_models, X_test, y_test):
    """Select the highest-F1 model and save it."""

    # Select based on F1-score for the bad-risk class.
    # This balances precision and recall for that class.
    best_row = results_df.sort_values(
        by="F1 Score",
        ascending=False,
    ).iloc[0]

    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]

    model_path = MODEL_DIR / "best_credit_scoring_model.joblib"

    joblib.dump(best_model, model_path)

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print("Selected model:", best_model_name)
    print(f"F1 Score: {best_row['F1 Score']:.4f}")
    print(f"Accuracy: {best_row['Accuracy']:.4f}")

    print(f"\nSaved model to: {model_path}")

    # Create a confusion matrix for the selected model.
    predictions = best_model.predict(X_test)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["Good Risk", "Bad Risk"],
    )

    display.ax_.set_title(
        f"Confusion Matrix - {best_model_name}"
    )

    plt.tight_layout()

    confusion_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.savefig(confusion_path, dpi=300)
    plt.close()

    print(f"Confusion matrix saved to: {confusion_path}")

    print("\nSample predictions:")
    print("0 = Good credit risk")
    print("1 = Bad credit risk")

    sample = X_test.head(5)
    sample_predictions = best_model.predict(sample)

    for index, prediction in zip(
        sample.index,
        sample_predictions,
    ):
        print(
            f"Applicant {index}: "
            f"{'Bad credit risk' if prediction == 1 else 'Good credit risk'}"
        )


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():
    """Run the complete credit scoring workflow."""

    df = load_dataset()

    X, y = prepare_data(df)

    # Keep the same proportion of each class in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    models = create_models(X)

    results_df, trained_models = train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    save_results(results_df)

    save_best_model(
        results_df,
        trained_models,
        X_test,
        y_test,
    )

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()