import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from model import HousePriceModel

# Load the data
data = pd.read_csv('house_prices.csv')
print("$1")
print("WElcome")
print("WElcome")

def test_example():
    assert True

# Split features and target
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = HousePriceModel()
model.train(X_train, y_train)

# Evaluate the model
train_mse = mean_squared_error(y_train, model.predict(X_train))
test_mse = mean_squared_error(y_test, model.predict(X_test))

print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")

# Save the model
model.save('house_price_model.joblib')
print("3333")