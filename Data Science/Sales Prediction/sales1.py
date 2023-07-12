import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load your sales data into a Pandas DataFrame
# Assuming your data is in a CSV file named 'sales_data.csv' with columns:
# 'AdvertisingExpenditure', 'TargetAudienceSegmentation', 'AdvertisingPlatform', and 'Sales'
data = pd.read_csv('sales_data.csv')

# Step 2: Prepare the data
X = data[['AdvertisingExpenditure', 'TargetAudienceSegmentation', 'AdvertisingPlatform']].values
y = data['Sales'].values

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the Gradient Boosting Regression model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 7: You can now use the trained model to make predictions for new data
# Example new data points with 'AdvertisingExpenditure', 'TargetAudienceSegmentation', and 'AdvertisingPlatform'
new_data = np.array([[1000, 1, 2], [500, 2, 1], [2000, 3, 3]])
new_predictions = model.predict(new_data)
print("New Predictions:")
print(new_predictions)
