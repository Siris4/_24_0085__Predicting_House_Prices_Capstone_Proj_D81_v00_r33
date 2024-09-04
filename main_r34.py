import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import skew

# Load the dataset
data = pd.read_csv(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\r00_env_START\boston.csv', index_col=0)

# Define the features (X) and the log-transformed target (y_log)
X = data.drop(columns=['PRICE'])  # All features except PRICE
y_log = np.log(data['PRICE'])     # Log-transformed PRICE

# Split the dataset into training and testing sets (80/20 split) with the same random state
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=10)

# Initialize and fit the Linear Regression model
model_log = LinearRegression()
model_log.fit(X_train, y_train_log)

# Predict the log-transformed prices for the training data
y_train_log_pred = model_log.predict(X_train)

# Calculate the residuals for the log-transformed prices
residuals_log = y_train_log - y_train_log_pred

# Calculate the mean and skew of the residuals using log-transformed prices
mean_residuals_log = residuals_log.mean()
skew_residuals_log = skew(residuals_log)

# Print the results to the console
print(f"Mean of residuals (log prices): {mean_residuals_log:.4f}")
print(f"Skewness of residuals (log prices): {skew_residuals_log:.4f}")
