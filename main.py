# Import Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Load Dataset

iris = load_iris()


# Display Dataset Information

print("=" * 50)
print("Feature Names:")
print(iris.feature_names)

print("\nTarget Names:")
print(iris.target_names)

print("\nFirst 5 Rows of Data:")
print(iris.data[:5])

print("\nFirst 5 Labels:")
print(iris.target[:5])


# Create DataFrame

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["Species"] = iris.target

print("\nDataFrame:")
print(df.head())


# Features & Target

X = iris.data
y = iris.target


# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# Create Model

model = DecisionTreeClassifier(random_state=42)


# Train Model

model.fit(X_train, y_train)


# Prediction

y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test)


# Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)


# Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))