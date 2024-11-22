import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Load the CSV files
population_growth_df = pd.read_csv('population_growth.csv')

unique_species_year_pairs_per_site = (
    population_growth_df.groupby('SiteID')
    .apply(lambda x: x[['CommonName', 'Year']].drop_duplicates().shape[0])
    .reset_index()
)

unique_species_year_pairs_per_site.columns = ['SiteID', 'UniqueSpeciesYearPairs']

# Filter for sites with more than 1000 unique species-year pairs
sites_over_1000 = unique_species_year_pairs_per_site[unique_species_year_pairs_per_site['UniqueSpeciesYearPairs'] > 1000]['SiteID']
sites_under_1000 = unique_species_year_pairs_per_site[unique_species_year_pairs_per_site['UniqueSpeciesYearPairs'] < 1000]['SiteID']

t1_sites = population_growth_df[population_growth_df['SiteID'].isin(sites_over_1000)]
t2_sites = population_growth_df[population_growth_df['SiteID'].isin(sites_under_1000)]

#TEST: ONLY MEASURE WITH SITE 2
site_2 = t1_sites[t1_sites['SiteID'] == 2]

# Separate features and target variable
X = site_2[['Year', 'WinterStage', 'ResStatus', 'Hostplant_Specialism_Score', 'HostCategory', 'Voltinism', 'WingSize_meanFemale', 'Wetland', 'Disturbed']]
y = site_2['GrowthRate']

# Impute missing values in 'meanLL' using median strategy
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=site_2[['Year', 'WinterStage', 'ResStatus', 
                                                          'Hostplant_Specialism_Score', 'HostCategory', 
                                                          'Voltinism', 'WingSize_meanFemale', 
                                                          'Wetland', 'Disturbed']].columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Train RandomForestRegressor
model = RandomForestRegressor(random_state = 0)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate and print metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training MAE:", train_mae)
print("Test MAE:", test_mae)
print("Training MSE:", train_mse)
print("Test MSE:", test_mse)
print("Training R-squared:", train_r2)
print("Test R-squared:", test_r2)

site_2 = site_2[['Year', 'WinterStage', 'ResStatus', 'Hostplant_Specialism_Score', 'HostCategory', 'Voltinism', 'WingSize_meanFemale', 'Wetland', 'Disturbed', 'GrowthRate']]

corr = site_2.corr()
print(corr['GrowthRate'].sort_values(ascending=False))

# Feature importance plot
importances = model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names)
feat_imp.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances in Random Forest")
plt.show()