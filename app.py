
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Used-Book Resale Pricing",

    page_icon="📚",
    layout="wide"
)

st.title("📚 Used-Book Resale Price Prediction")

st.write(
    "This application uses Machine Learning and Decision Tree "
    "Regression to estimate the resale price of a used book."
)

# Load dataset
data = pd.read_csv("used_books.csv")

# Load trained model
with open("book_price_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model_columns.pkl", "rb") as file:
    model_columns = pickle.load(file)

st.divider()

# Book details
st.header("Enter Book Details")

col1, col2 = st.columns(2)

with col1:
    original_price = st.number_input(
        "Original Book Price (₹)",
        min_value=1,
        max_value=100000,
        value=600
    )

    book_age = st.number_input(
        "Book Age (Years)",
        min_value=0,
        max_value=100,
        value=2
    )

with col2:
    condition = st.selectbox(
        "Book Condition",
        ["Excellent", "Good", "Average", "Poor"]
    )

    subject = st.selectbox(
        "Book Subject",
        [
            "Programming",
            "Mathematics",
            "Science",
            "Literature",
            "Commerce"
        ]
    )

if st.button("Predict Resale Price"):
    input_data = pd.DataFrame([{
        "Original_Price": original_price,
        "Book_Age": book_age,
        "Condition": condition,
        "Subject": subject
    }])

    input_data = pd.get_dummies(
        input_data,
        columns=["Condition", "Subject"]
    )

    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    predicted_price = model.predict(input_data)[0]

    st.success(
        f"Estimated Resale Price: ₹{predicted_price:.2f}"
    )

    st.info(
        "This is an estimate from a model trained on synthetic data. "
        "Actual market prices may differ."
    )

st.divider()

# Model performance
st.header("📊 Model Performance")

X = data[
    ["Original_Price", "Book_Age", "Condition", "Subject"]
]

y = data["Resale_Price"]

X = pd.get_dummies(
    X,
    columns=["Condition", "Subject"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

predictions = model.predict(
    X_test.reindex(columns=model_columns, fill_value=0)
)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Total Books", len(data))
metric2.metric("Mean Absolute Error", f"₹{mae:.2f}")
metric3.metric("R² Score", f"{r2:.4f}")

# Actual vs Predicted graph
st.header("📈 Actual vs Predicted Book Prices")

fig, ax = plt.subplots(figsize=(4, 2.5))

ax.scatter(
    y_test,
    predictions,
    alpha=0.7,
    label="Predicted Prices"
)

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

ax.set_xlabel("Actual Resale Price (₹)")
ax.set_ylabel("Predicted Resale Price (₹)")
ax.set_title("Actual vs Predicted Prices")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig,use_container_width=False)
plt.close(fig)


st.caption(
    "The dashed line represents perfect predictions. "
    "Points closer to this line indicate more accurate predictions."
)

# Dataset
st.header("📚 Used-Book Dataset")

st.dataframe(data.head(20), use_container_width=True)

csv_data = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Complete Dataset",
    data=csv_data,
    file_name="used_books.csv",
    mime="text/csv"
)

st.caption(
    "Demonstration model trained on synthetic book prices."
)
