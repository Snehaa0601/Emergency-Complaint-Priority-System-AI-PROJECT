import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
data = pd.read_csv("emergency_complaints_dataset.csv")

print("Dataset loaded successfully")
print(data.head())


# 2. Create Label Encoders
category_encoder = LabelEncoder()
priority_encoder = LabelEncoder()


# 3. Convert complaint category into numbers
data["complaint_category"] = category_encoder.fit_transform(
    data["complaint_category"]
)


# 4. Convert priority into numbers
data["priority"] = priority_encoder.fit_transform(
    data["priority"]
)


# 5. Select input features
X = data[
    [
        "complaint_category",
        "severity",
        "people_affected",
        "response_urgency"
    ]
]


# 6. Select target
y = data["priority"]


# 7. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 8. Create Decision Tree model
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)


# 9. Train model
model.fit(X_train, y_train)


# 10. Make predictions
y_pred = model.predict(X_test)


# 11. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", accuracy * 100, "%")


# 12. Display classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=priority_encoder.classes_,
        zero_division=0
    )
)


# 13. Save model and encoders
with open("priority_model.pkl", "wb") as file:

    pickle.dump(
        (
            model,
            category_encoder,
            priority_encoder
        ),
        file
    )


print("\nModel saved successfully as priority_model.pkl")
