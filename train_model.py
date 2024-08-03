from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import pandas as pd
import data_preprocessing
# Load processed data
X = data_preprocessing.X
y = data_preprocessing.y
file_path = 'D:/Mtech/Semester3/MLOPS/Assignment/mlops_assignment_1_group_56/train_output'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
output_file_path = 'D:/Mtech/Semester3/MLOPS/Assignment/mlops_assignment_1_group_56/test_output'
# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(X_train)
print(y_train)
print(X_train.columns.tolist())
X_train.to_csv(file_path+'/X_train.csv')
y_train.to_csv(file_path+'/Y_train.csv')
X_test.to_csv(output_file_path+'/X_test.csv')
y_test.to_csv(output_file_path+'/Y_test.csv')

# Save the model
dump(model, 'D:/Mtech/Semester3/MLOPS/Assignment/mlops_assignment_1_group_56/models/model.joblib')
