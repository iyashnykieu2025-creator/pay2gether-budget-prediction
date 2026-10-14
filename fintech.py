from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Use a fixed seed so the same dataset is created every time.
RANDOM_SEED = 42
NUMBER_OF_TRIPS = 1000

rng = np.random.default_rng(RANDOM_SEED)

# Create three pieces of information known before a trip begins.
number_of_members = rng.integers(2, 11, NUMBER_OF_TRIPS)
trip_length_days = rng.integers(1, 15, NUMBER_OF_TRIPS)
planned_budget_per_person = np.round(
    rng.uniform(150, 1800, NUMBER_OF_TRIPS), 2
)

# Longer trips and lower budgets have a greater chance of going over budget.
# Larger groups receive a small saving because they can share some costs.
estimated_cost_per_person = (
    250
    + (70 * trip_length_days)
    - (12 * number_of_members)
)
random_difference = rng.normal(0, 220, NUMBER_OF_TRIPS)
actual_cost_per_person = estimated_cost_per_person + random_difference

# 1 means the group exceeded its planned budget; 0 means it stayed within it.
exceeded_budget = (
    actual_cost_per_person > planned_budget_per_person
).astype(int)

data = pd.DataFrame(
    {
        "number_of_members": number_of_members,
        "trip_length_days": trip_length_days,
        "planned_budget_per_person": planned_budget_per_person,
        "exceeded_budget": exceeded_budget,
    }
)

output_directory = Path(__file__).resolve().parent
excel_path = output_directory / "fintech.xlsx"
model_path = output_directory / "fintech.joblib"

data.to_excel(excel_path, index=False)

X = data[
    [
        "number_of_members",
        "trip_length_days",
        "planned_budget_per_person",
    ]
]
y = data["exceeded_budget"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y,
)

# Scaling and logistic regression are kept together in one saved model.
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "logistic_regression",
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        ),
    ]
)

model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

print("Training classification report")
print(classification_report(y_train, train_predictions))

print("Test classification report")
print(classification_report(y_test, test_predictions))

joblib.dump(model, model_path)

print(f"Dataset saved to: {excel_path}")
print(f"Model saved to: {model_path}")

