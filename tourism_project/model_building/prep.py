import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
from datasets import load_dataset

# Initialize Hugging Face API with token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 2: Load the dataset from Hugging Face
dataset = load_dataset("hkbindhu/Tourism-Package-Prediction") 
data = dataset['train'].to_pandas()
print("✅ Dataset loaded successfully.")

# Step 3: Clean the dataset

# Drop irrelevant columns
columns_to_drop = [
    "CustomerID", "DurationOfPitch", "NumberOfFollowups",
    "ProductPitched", "PitchSatisfactionScore"
]
data.drop(columns=columns_to_drop, inplace=True)

# Fill missing numeric values with mean
numeric_features = [
    'Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 
    'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]
data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())

# Fill missing categorical values with mode
categorical_columns = ['Gender', 'TypeofContact', 'Occupation', 'MaritalStatus', 'Designation']
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Label encode 'Gender'
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])  # Male=1, Female=0

# One-hot encode selected categorical columns
one_hot_columns = ['TypeofContact', 'Occupation', 'MaritalStatus', 'Designation']
data = pd.get_dummies(data, columns=one_hot_columns, drop_first=True)

# Define features and target
target = 'Exited'  # Adjust if your target is different
X = data.drop(columns=[target])  # Features
y = data[target]                 # Target

print("✅ Data cleaned: columns dropped, missing values handled, encodings applied.")


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

Xtrain[target] = ytrain.values
Xtest[target] = ytest.values

