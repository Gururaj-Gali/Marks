import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load data
data = pd.read_csv("data.csv")

X = data[['Hours']]
y = data['Marks']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved")
