
import pandas as pd
import numpy as np

np.random.seed(42)

number_of_books = 500

original_prices = np.random.randint(100, 2001, number_of_books)
book_ages = np.random.randint(0, 11, number_of_books)

conditions = np.random.choice(
    ["Excellent", "Good", "Average", "Poor"],
    number_of_books
)

subjects = np.random.choice(
    ["Programming", "Mathematics", "Science", "Literature", "Commerce"],
    number_of_books
)

condition_factors = {
    "Excellent": 0.85,
    "Good": 0.70,
    "Average": 0.50,
    "Poor": 0.30
}

resale_prices = []

for i in range(number_of_books):
    condition_factor = condition_factors[conditions[i]]
    age_factor = max(0.20, 1 - book_ages[i] * 0.07)

    price = original_prices[i] * condition_factor * age_factor
    noise = np.random.uniform(-20, 20)

    resale_prices.append(round(max(20, price + noise), 2))

data = pd.DataFrame({
    "Original_Price": original_prices,
    "Book_Age": book_ages,
    "Condition": conditions,
    "Subject": subjects,
    "Resale_Price": resale_prices
})

data.to_csv("used_books.csv", index=False)

print("Dataset created successfully!")
print("File saved as used_books.csv")
print(data.head())