import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

# Predict the log-transformed prices for the training and testing sets
y_train_log_pred = model.predict(X_train)
y_test_log_pred = model.predict(X_test)

# Calculate the R-squared value for both the training and test sets
r2_train_log = r2_score(y_train_log, y_train_log_pred)
r2_test_log = r2_score(y_test_log, y_test_log_pred)

# Output the R-squared values
print(f"R-squared value on training data (log prices): {r2_train_log:.4f}")
print(f"R-squared value on test data (log prices): {r2_test_log:.4f}")
