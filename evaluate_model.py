from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import load
import pandas as pd

test_path = 'test_output'
output_path = 'output'

# Load test data and model
X_test = pd.read_csv(test_path+'/X_test.csv')

# Remove the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in X_test.columns:
    X_test = X_test.drop(columns=['Unnamed: 0'])

model = load('models/model.joblib')

# Make predictions
y_pred = model.predict(X_test)
print(y_pred)
# Convert y_pred to DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=['Prediction'])

# Combine X_test and y_pred_df
combined_df = pd.concat([X_test, y_pred_df], axis=1)
print(combined_df)
# Save the combined DataFrame to a CSV file
combined_df.to_csv(output_path+'/final_predicted_output.csv', index=False)

