import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("Students_Marks_Dataset.csv")

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Assignments",
    "Sleep_Hours",
    "Past_Performance"
]

TARGET = "Final_Marks"

# 🔑 Drop rows with missing TARGET
df = df.dropna(subset=[TARGET])

X = df[FEATURES]
y = df[TARGET]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline (Features only)
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "model/marks_model.pkl")

print("✅ Model trained and saved successfully")
