
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("used_books.csv")

# Select input features
X = data[
    ["Original_Price", "Book_Age", "Condition", "Subject"]
]

# Target variable
y = data["Resale_Price"]

# Convert text columns into numeric columns
X = pd.get_dummies(X, columns=["Condition", "Subject"])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree Regression model
model = DecisionTreeRegressor(
    max_depth=8,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Evaluate model performance
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model trained successfully!")
print(f"Mean Absolute Error: Rs. {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# Save trained model
with open("book_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save model input columns
with open("model_columns.pkl", "wb") as file:
    pickle.dump(X.columns.tolist(), file)

print("Model saved successfully!")
print("Created: book_price_model.pkl")
print("Created: model_columns.pkl")
