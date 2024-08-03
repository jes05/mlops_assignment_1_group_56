from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import pandas as pd
import data_preprocessing
import mlflow
# Load processed data
X = data_preprocessing.X
y = data_preprocessing.y
file_path = 'train_output'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
# Set up MLflow tracking
mlflow.set_experiment(experiment_id="559978265238575036")
print("getting tracking uri")
print(mlflow.get_tracking_uri())
print("setting artifact uri")
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.autolog()
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Log the model
    mlflow.sklearn.log_model(model, 'model')
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric('accuracy', accuracy)

    mlflow.log_param('n_estimators', model.n_estimators)
output_file_path = 'test_output'
"""
OLD VERSION ---- 
# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(X_train)
print(y_train)
print(X_train.columns.tolist())
"""
X_train.to_csv(file_path+'/X_train.csv')
y_train.to_csv(file_path+'/Y_train.csv')
X_test.to_csv(output_file_path+'/X_test.csv')
y_test.to_csv(output_file_path+'/Y_test.csv')

# Save the model
dump(model, 'models/model.joblib')
