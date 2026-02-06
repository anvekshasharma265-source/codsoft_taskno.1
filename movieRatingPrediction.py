# Movie Rating Prediction using Python
# Regression-based Machine Learning Project

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("data/movies.csv")

# Separate features and target
X = data.drop("rating", axis=1)
y = data["rating"]

# Define feature types
categorical_features = ["genre", "director", "actor"]
numerical_features = ["year"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
print("Model Performance")
print("-----------------")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Example prediction
example_movie = pd.DataFrame({
    "genre": ["Action"],
    "director": ["Nolan"],
    "actor": ["DiCaprio"],
    "year": [2024]
})

predicted_rating = pipeline.predict(example_movie)
print("\nPredicted Rating for Example Movie:", round(predicted_rating[0], 2))
