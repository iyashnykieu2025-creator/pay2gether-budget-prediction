# Pay2Gether Budget Prediction

This project is a machine-learning web application for Pay2Gether. It predicts whether a group trip is likely to exceed its planned budget. This can help groups choose a more realistic budget before their trip.

## Machine-Learning Model

The application uses logistic regression for binary classification.

The target variable is `exceeded_budget`:

- `1` means the group is likely to exceed its budget.
- `0` means the group is likely to stay within its budget.

The model uses three input variables:

- `number_of_members`: number of people in the group
- `trip_length_days`: length of the trip in days
- `planned_budget_per_person`: planned budget for each person

The model was trained with 1,000 rows of synthetic data generated using NumPy with a fixed random seed.

## Live Application

The application is available at:

https://ivanna23.pythonanywhere.com

Users can enter the three input variables and receive a prediction and the probability of exceeding the budget.

## API Endpoint

**Method:** POST

**URL:**

```text
https://ivanna23.pythonanywhere.com/api/predict
```

### Example request

```bash
curl -X POST -H "Content-Type: application/json" -d '{"number_of_members":3,"trip_length_days":12,"planned_budget_per_person":250}' https://ivanna23.pythonanywhere.com/api/predict
```

### Example response

```json
{
  "prediction": 1,
  "probability_of_exceeding_budget": 0.9976
}
```

## Project Files

- `fintech.py`: generates the data and trains the model
- `fintech.xlsx`: synthetic dataset
- `fintech.joblib`: trained model
- `flask_app.py`: web application and API
- `templates/index.html`: HTML form
- `static/style.css`: page design
- `requirements.txt`: required Python packages

## Setup Instructions

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Train and save the model:

```bash
python3 fintech.py
```

3. Run the Flask application:

```bash
flask --app flask_app run
```