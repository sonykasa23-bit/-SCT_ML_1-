#  Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#  Load the training data
df = pd.read_csv('train.csv')

# Explore the data
print("First 5 rows of data:")
print(df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].head())

#  Check for missing values
print("\nMissing values in selected columns:")
print(df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].isnull().sum())

#  Define features and label
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]  # Input features
y = df['SalePrice']  # Target label

# Fill any missing values with average (just in case)
X = X.fillna(X.mean())

#  Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#  Create the Linear Regression model
model = LinearRegression()

#  Train the model on training data
model.fit(X_train, y_train)

#  Predict on test data
y_pred = model.predict(X_test)

#  Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("\nRoot Mean Squared Error (RMSE):", rmse)

#  Visualize Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # perfect prediction line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()