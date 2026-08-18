import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR

# 1. Past data (e.g., Month numbers 1 to 6, and fuel demand in liters)
W = np.array([[1], [2], [3], [4], [5], [6]])  # Months
y = np.array([12000, 12500, 13000, 13300, 13800, 14200])  # Demand in Liters

# 2. Create and train the model
model = LR()
model.fit(W, y)

# 3. Forecast demand for the next 3 months (Months 7, 8, and 9)
X_future = np.array([[7], [8], [9]])
y_pred = model.predict(X_future)

# 4. Print results
for month, demand in zip(X_future.ravel(), y_pred):
    print(f"Month {month}: Predicted Demand = {demand:.2f} Liters")

# 5. Plot past data and future forecast
plt.scatter(W, y, color='blue', label='Past Demand')
plt.plot(X_future, y_pred, color='red', linestyle='--', label='Forecast Demand')
plt.xlabel('Month')
plt.ylabel('Fuel Demand (Liters)')
plt.legend()
plt.show()
