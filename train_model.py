import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
file_path = "Dys_Cleaned1.csv"  # Ensure this file is in the same directory
dyslexia_data = pd.read_csv(file_path)

# Define feature names (excluding 'Dyslexia')
FEATURE_NAMES = ['Gender', 'Age', 'Clicks4', 'Hits4', 'Misses4', 'Score4', 'Accuracy4',
                 'Missrate4', 'Clicks12', 'Hits12', 'Misses12', 'Score12', 'Accuracy12',
                 'Missrate12', 'Clicks26', 'Hits26', 'Misses26', 'Score26', 'Accuracy26',
                 'Missrate26', 'Clicks27', 'Hits27', 'Misses27', 'Score27', 'Accuracy27',
                 'Missrate27']

# Split data
X = dyslexia_data[FEATURE_NAMES]
y = dyslexia_data['Dyslexia']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "dyslexia_model.pkl")

print("✅ Model trained and saved as dyslexia_model.pkl")
