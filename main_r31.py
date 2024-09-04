import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\r00_env_START\boston.csv', index_col=0)

# Define the features (X) and the log-transformed target (y_log)
X = data.drop(columns=['PRICE'])  # All features except PRICE
y_log = np.log(data['PRICE'])     # Log-transformed PRICE

# Split the dataset into training and testing sets (80/20 split) with the same random state
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=10)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train_log)

# Predict the log-transformed prices for the training data
y_train_log_pred = model.predict(X_train)

# Calculate the residuals for the training data
residuals_log = y_train_log - y_train_log_pred

# Scatter plot of actual vs. predicted log prices
plt.figure(figsize=(10, 6))
plt.scatter(y_train_log, y_train_log_pred, alpha=0.5)
plt.title('Actual vs. Predicted Log-Transformed Prices')
plt.xlabel('Actual Log(Prices)')
plt.ylabel('Predicted Log(Prices)')
plt.grid(True)
plt.savefig(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\actual_vs_predicted_log_prices.png')

# Scatter plot of residuals vs. predicted log prices
plt.figure(figsize=(10, 6))
plt.scatter(y_train_log_pred, residuals_log, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Predicted Log-Transformed Prices')
plt.xlabel('Predicted Log(Prices)')
plt.ylabel('Residuals')
plt.grid(True)
plt.savefig(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\residuals_vs_predicted_log_prices.png')

