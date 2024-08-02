from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import pandas as pd

# Load processed data
X = pd.read_csv('path/to/X.csv')
y = pd.read_csv('path/to/y.csv')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
dump(model, 'models/model.joblib')
