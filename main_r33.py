import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

# Calculate the residuals for both original and log-transformed prices
residuals = y_train - y_train_pred
residuals_log = y_train_log - y_train_log_pred

# Create 2x2 subplot for comparing original and log-transformed price regression outcomes with specified colors
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Scatter plot of actual vs. predicted prices (original) with indigo color
axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5, color='indigo')
axes[0, 0].set_title('Actual vs. Predicted Prices (Original)')
axes[0, 0].set_xlabel('Actual Prices')
axes[0, 0].set_ylabel('Predicted Prices')

# Scatter plot of residuals vs. predicted prices (original) with indigo color
axes[0, 1].scatter(y_train_pred, residuals, alpha=0.5, color='indigo')
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_title('Residuals vs. Predicted Prices (Original)')
axes[0, 1].set_xlabel('Predicted Prices')
axes[0, 1].set_ylabel('Residuals')

# Scatter plot of actual vs. predicted log prices with navy color
axes[1, 0].scatter(y_train_log, y_train_log_pred, alpha=0.5, color='navy')
axes[1, 0].set_title('Actual vs. Predicted Log-Transformed Prices')
axes[1, 0].set_xlabel('Actual Log(Prices)')
axes[1, 0].set_ylabel('Predicted Log(Prices)')

# Scatter plot of residuals vs. predicted log prices with navy color
axes[1, 1].scatter(y_train_log_pred, residuals_log, alpha=0.5, color='navy')
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title('Residuals vs. Predicted Log-Transformed Prices')
axes[1, 1].set_xlabel('Predicted Log(Prices)')
axes[1, 1].set_ylabel('Residuals')

# Adjust layout and show the plots
plt.tight_layout()
plt.savefig(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\comparison_regression_outcomes_colored.png')

