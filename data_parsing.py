import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

#Population data to be fixed
pop_data = pd.read_csv("/Users/noahh/Downloads/doi_10_5061_dryad_cf78420__v20190625/Dryad_butterfly_trends/data/allpops.csv")
#Species' trait data to be encoded
trait_data = pd.read_csv("/Users/noahh/Downloads/doi_10_5061_dryad_cf78420__v20190625/Dryad_butterfly_trends/data/speciestraits.csv")


# Impute missing values in 'meanLL' using median strategy
imputer = SimpleImputer(strategy='median')
pop_data['meanLL'] = imputer.fit_transform(pop_data[['meanLL']])

#Sort the data by SiteID, CommonName, and Year
pop_data = pop_data.sort_values(by=['SiteID', 'CommonName', 'Year'])

# Averaging the Index values for each SiteID, CommonName, and Year
# This is because there are two analysis methods used - ukbms and gampred
pop_data_unique = pop_data.groupby(['SiteID', 'CommonName', 'Year'])['Index'].mean().reset_index()

#Calculate the percent change (growth rate) for each group
pop_data_unique['GrowthRate'] = pop_data.groupby(['SiteID', 'CommonName'])['Index'].pct_change() * 100

print(pop_data_unique[['SiteID', 'CommonName', 'Year', 'Index', 'GrowthRate']].head())
