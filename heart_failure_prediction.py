import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
heart_failure = pd.read_csv('heart_failure_clinical_records.csv')

# Display the first 5 rows of the dataset
print(heart_failure.head())

# Display the shape of the dataset
print(heart_failure.shape)

# Display the information of the dataset
print(heart_failure.info())

# Max and minimum values
print('Max values:', heart_failure.max())
print('Min values:', heart_failure.min())

# Define features (X) and target (y)
X = heart_failure[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]
y = heart_failure['DEATH_EVENT']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

# Training the model on the training set
classifier = RandomForestClassifier(random_state=1000)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Evaluating the performance of the classification model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Displaying actual vs. predicted values
results = pd.DataFrame({'Death Event': y_test, 'Predicted Death Event': y_pred})
print(results.head(20))

# Load new data to predict
heart_failure_data = pd.read_csv('heart_failure_data.csv')

# Predict heart failure death event
predicted_heart_failure = classifier.predict(heart_failure_data)

# Display the failure death event
predicted_heart_failure_df = pd.DataFrame({'Predicted Heart Failure Data': predicted_heart_failure})
print(predicted_heart_failure_df.head(100))

# Create plot
plt.figure(figsize=(10, 6))
sns.countplot(x='Predicted Heart Failure Data', data=predicted_heart_failure_df)
plt.title('Heart Failure Prediction')
plt.show()