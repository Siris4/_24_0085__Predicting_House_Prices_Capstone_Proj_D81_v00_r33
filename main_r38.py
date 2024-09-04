import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\r00_env_START\boston.csv', index_col=0)

# Drop the target variable 'PRICE' to get only the features
features = data.drop(columns=['PRICE'], axis=1)

# Calculate the average values for each feature
average_vals = features.mean().values

# Create a DataFrame for the average property stats
property_stats = pd.DataFrame(data=average_vals.reshape(1, len(features.columns)), columns=features.columns)

# Initialize and fit the Linear Regression model for the log-transformed prices
y_log = np.log(data['PRICE'])  # Log-transformed target variable (PRICE)
model_log = LinearRegression()
model_log.fit(features, y_log)

# Get the coefficients from the model
coefficients = model_log.coef_

# Calculate the log price estimate using the average property stats and the model coefficients
log_price_estimate = np.dot(property_stats.values, coefficients) + model_log.intercept_

# Convert the log price to an actual price estimate using the exponential function
dollar_price_estimate = np.exp(log_price_estimate)

# Print the results
print(f"Log Price Estimate: {log_price_estimate[0]:.4f}")
print(f"Price Estimate in Dollars: ${dollar_price_estimate[0]:,.2f}")
