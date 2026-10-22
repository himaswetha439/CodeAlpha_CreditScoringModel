
from pathlib import Path

import joblib
import pandas as pd


# Locate the project directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "best_credit_scoring_model.joblib"
)


def main():
    # Check whether the trained model exists.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run main.py first."
        )

    # Load the saved machine-learning pipeline.
    model = joblib.load(MODEL_PATH)

    # Example applicant data.
    # These values are illustrative, not real applicant data.
    applicant = {
        "checking_account": "A11",
        "duration_months": 12,
        "credit_history": "A34",
        "purpose": "A43",
        "credit_amount": 2500,
        "savings_account": "A61",
        "employment_duration": "A73",
        "installment_rate": 2,
        "personal_status_sex": "A93",
        "other_debtors": "A101",
        "residence_since": 3,
        "property": "A121",
        "age_years": 30,
        "other_installment_plans": "A143",
        "housing": "A152",
        "existing_credits": 1,
        "job": "A173",
        "dependents": 1,
        "telephone": "A191",
        "foreign_worker": "A201",
    }

    # Convert the applicant data into a DataFrame.
    applicant_df = pd.DataFrame([applicant])

    # Predict the applicant's credit-risk category.
    prediction = model.predict(applicant_df)[0]

    # Display the result.
    print("\nCredit Scoring Model")
    print("-" * 35)

    if prediction == 1:
        print("Predicted category: Bad credit risk")
    else:
        print("Predicted category: Good credit risk")

    # Display the model's estimated probability.
    probabilities = model.predict_proba(applicant_df)[0]

    print(
        f"Estimated good-risk probability: "
        f"{probabilities[0]:.2%}"
    )

    print(
        f"Estimated bad-risk probability: "
        f"{probabilities[1]:.2%}"
    )

    print("\nNote: This is an educational model,")
    print("not a real-world lending recommendation.")


if __name__ == "__main__":
    main()

