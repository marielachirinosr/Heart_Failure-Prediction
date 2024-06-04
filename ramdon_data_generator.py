import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(0)  

# Define the number of samples
num_samples = 100000

# Generate features
age = np.random.randint(40, 95, size=num_samples)
anaemia = np.random.randint(0, 1, size=num_samples) 
creatinine_phosphokinase = np.random.randint(20, 8000, size=num_samples)
diabetes = np.random.randint(0, 1, size=num_samples)
ejection_fraction = np.random.randint(14, 80, size=num_samples)
high_blood_pressure = np.random.randint(0, 1, size=num_samples)
platelets = np.random.randint(250000, 850000, size=num_samples)
serum_creatinine = np.random.uniform(0.5, 9.4, size=num_samples)
serum_sodium = np.random.randint(113, 148, size=num_samples)
sex = np.random.randint(0, 1, size=num_samples)
smoking = np.random.randint(0, 1, size=num_samples)
time = np.random.randint(4, 285, size=num_samples)


# Create a DataFrame
heart_failure_data = pd.DataFrame({'age': age, 'anaemia': anaemia, 'creatinine_phosphokinase': creatinine_phosphokinase, 'diabetes': diabetes, 'ejection_fraction': ejection_fraction, 'high_blood_pressure': high_blood_pressure, 'platelets': platelets, 'serum_creatinine': serum_creatinine, 'serum_sodium': serum_sodium, 'sex': sex, 'smoking': smoking, 'time': time,})

# Save the DataFrame to a CSV file
heart_failure_data.to_csv('heart_failure_data.csv', index=False)

# Display the first few rows of the dataset
print(heart_failure_data.head(100))
