import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder


try:
    data = pd.read_csv('Crop_recommendationV2.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset file 'Crop_recommendationV2.csv' not found!")
    exit()
except pd.errors.EmptyDataError:
    print("Error: Dataset is empty!")
    exit()
except pd.errors.ParserError:
    print("Error: Dataset file is corrupt or improperly formatted!")
    exit()


print("Dataset Columns:", list(data.columns))


possible_targets = ['crop_health', 'label', 'growth_stage']
target_column = None

for col in possible_targets:
    if col in data.columns:
        target_column = col
        break

if target_column is None:
    print("Error: No valid target variable found in dataset!")
    exit()

print(f"Using '{target_column}' as the target variable.")


if data[target_column].dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(data[target_column])  
else:
    y = data[target_column]


numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
exclude_columns = [target_column]  # Exclude target variable from features
feature_columns = [col for col in numerical_columns if col not in exclude_columns]

if not feature_columns:
    print("Error: No numerical features available for training!")
    exit()

X = data[feature_columns]


X.fillna(X.median(), inplace=True)


try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dataset split into training and testing sets.")
except ValueError as e:
    print(f"Error during train-test split: {e}")
    exit()


try:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed successfully!")
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

try:
    y_pred = model.predict(X_test)
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()


try:
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
except Exception as e:
    print(f"Error during evaluation: {e}")
try:
    joblib.dump(model, 'farm_assistant_model.pkl')
    print("Model saved successfully as 'farm_assistant_model.pkl'.")
except Exception as e:
    print(f"Error while saving the model: {e}")
