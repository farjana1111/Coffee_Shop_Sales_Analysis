import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ✅ Define base directory relative to scripts folder
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
file_path = os.path.join(BASE_DIR, "coffee_shop_engineered.csv")

# 1. Load processed dataset
df = pd.read_csv(file_path)

# 2. Define features and target
X = df[['transaction_qty', 'unit_price', 'hour', 'month']]
y = df['total_amount']

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# 5. Predict on test set
y_pred = reg.predict(X_test)

# 6. Evaluate model (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

# 7. Show coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': reg.coef_
})
print("\nModel Coefficients:")
print(coefficients)
