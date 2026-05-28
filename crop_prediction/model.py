import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Read dataset
data = pd.read_csv("crop_recommendation.csv")

# Input columns
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Output column
y = data['label']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")