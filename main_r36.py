import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\r00_env_START\boston.csv', index_col=0)

# Define the features (X) and the targets for both original and log-transformed prices
X = data.drop(columns=['PRICE'])  # All features except PRICE
y = data['PRICE']                 # Original PRICE
y_log = np.log(data['PRICE'])     # Log-transformed PRICE

# Split the dataset into training and testing sets (80/20 split) with the same random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=10)

# Initialize and fit the Linear Regression model for both original and log-transformed prices
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)

model_log = LinearRegression()
model_log.fit(X_train_log, y_train_log)
y_train_log_pred = model_log.predict(X_train_log)

# Predict on the test data for both original and log-transformed prices
y_test_pred = model.predict(X_test)  # For original prices
y_test_log_pred = model_log.predict(X_test_log)  # For log-transformed prices

# Calculate the R-squared values for both models
r2_test_original = r2_score(y_test, y_test_pred)
r2_test_log = r2_score(y_test_log, y_test_log_pred)

# Print the R-squared values for comparison
print(f"R-squared value on test data (original prices): {r2_test_original:.4f}")
print(f"R-squared value on test data (log-transformed prices): {r2_test_log:.4f}")
