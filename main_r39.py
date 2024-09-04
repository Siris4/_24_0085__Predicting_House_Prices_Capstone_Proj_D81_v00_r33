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

# Define Property Characteristics
next_to_river = True
nr_rooms = 8
students_per_classroom = 20
distance_to_town = 5
pollution = data.NOX.quantile(q=0.75)  # high
amount_of_poverty = data.LSTAT.quantile(q=0.25)  # low

# Modify the specific property characteristics as per the challenge
property_stats['CHAS'] = 1 if next_to_river else 0  # Next to the river (1) or not (0)
property_stats['RM'] = nr_rooms  # Number of rooms
property_stats['PTRATIO'] = students_per_classroom  # Students per classroom
property_stats['DIS'] = distance_to_town  # Distance to employment center
property_stats['NOX'] = pollution  # Pollution level (high, using 75th percentile)
property_stats['LSTAT'] = amount_of_poverty  # Poverty level (low, using 25th percentile)

# Initialize and fit the Linear Regression model for the log-transformed prices
y_log = np.log(data['PRICE'])  # Log-transformed target variable (PRICE)
model_log = LinearRegression()
model_log.fit(features, y_log)

# Get the coefficients from the model
coefficients = model_log.coef_

# Calculate the log price estimate using the modified property stats and the model coefficients
log_price_estimate = np.dot(property_stats.values, coefficients) + model_log.intercept_

# Convert the log price to an actual price estimate using the exponential function
dollar_price_estimate = np.exp(log_price_estimate)

# Print the results
print(f"Log Price Estimate: {log_price_estimate[0]:.4f}")
print(f"Price Estimate in Dollars: ${dollar_price_estimate[0]:,.2f}")
