from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import load
import pandas as pd
import mlflow

test_path = 'test_output'
output_path = 'final_output'
mlflow.set_experiment(experiment_id="559978265238575036")
print(mlflow.get_tracking_uri())
mlflow.set_tracking_uri("https://github.com/jes05/mlops_assignment_1_group_56/tree/51185fb1b11352f931a6456bfb0f32fa9cd7864e/mlruns")
mlflow.autolog()


# Load test data and model
X_test = pd.read_csv(test_path+'/X_test.csv')

# Remove the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in X_test.columns:
    X_test = X_test.drop(columns=['Unnamed: 0'])

model = load('models/model.joblib')

# Make predictions
y_pred = model.predict(X_test)
print(y_pred)
with mlflow.start_run() as run:
    accuracy = accuracy_score(y_pred, model.predict(X_test))
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('n_estimators', model.n_estimators)
# Convert y_pred to DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=['Prediction'])

# Combine X_test and y_pred_df
combined_df = pd.concat([X_test, y_pred_df], axis=1)
print(combined_df)
# Save the combined DataFrame to a CSV file
combined_df.to_csv(output_path+'/final_predicted_output.csv', index=False)
combined_df.to_csv('output/final_predicted_output.csv', index=False)

