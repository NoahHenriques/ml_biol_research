import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

#Load your dataset
data = pd.read_csv('butterfly_data.csv')

# Handle missing values
data = data.d

# Encode categorical variables
data = pd.get_dummies(data, columns=['color_pattern'])

# Separate features and target variable
X = data.drop('population_change', axis=1)
y = data['population_change']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate model
score = model.score(X_test, y_test)
print(f'R-squared Score: {score}')

importances = model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names)
feat_imp.nlargest(10).plot(kind='barh')
plt.show()