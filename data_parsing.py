import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

#Population data to be fixed
pop_data = pd.read_csv("/Users/noahh/Downloads/doi_10_5061_dryad_cf78420__v20190625/Dryad_butterfly_trends/data/allpops.csv")
#Species' trait data to be encoded
trait_data = pd.read_csv("/Users/noahh/Downloads/doi_10_5061_dryad_cf78420__v20190625/Dryad_butterfly_trends/data/speciestraits.csv")

# Averaging the Index values for each SiteID, CommonName, and Year
# This is because there are two analysis methods used - ukbms and gampred
pop_data_unique_change = pop_data_unique = pop_data.groupby(['SiteID', 'CommonName', 'Year'])['Index'].mean().reset_index()

#Calculate the percent change (growth rate) for each group
pop_data_unique_change['GrowthRate'] = pop_data_unique.groupby(['SiteID', 'CommonName'])['Index'].pct_change()
pop_data_unique_change = pop_data_unique_change.dropna(subset=['GrowthRate'])

#Sort the data by SiteID, CommonName, and Year and drop Index
pop_data_unique_change = pop_data_unique_change.sort_values(by=['CommonName', 'SiteID', 'Year'])
pop_data_unique_change.drop("Index", axis=1, inplace=True)

#Import the species' traits and remove unwanted data
species_traits_df = pd.read_csv('/Users/noahh/Downloads/doi_10_5061_dryad_cf78420__v20190625/Dryad_butterfly_trends/data/speciestraits.csv')
species_traits_df.drop(["CombinedLatin", "N", "Total", "estimate", "std.error", "p.value", "Range_Num"], axis=1, inplace=True)

# Merge the dataframes on 'CommonName'
data = pd.merge(pop_data_unique_change, species_traits_df, on='CommonName', how='inner')

# Count unique species-year pairs per site with updated method to avoid warning
unique_species_year_pairs_per_site = (
    data.groupby('SiteID')
    .apply(lambda x: x[['CommonName', 'Year']].drop_duplicates().shape[0])
    .reset_index()
)
unique_species_year_pairs_per_site.columns = ['SiteID', 'UniqueSpeciesYearPairs']

# Filter for sites with more than 500 unique species-year pairs
sites_over_500 = unique_species_year_pairs_per_site[unique_species_year_pairs_per_site['UniqueSpeciesYearPairs'] > 500]['SiteID']

processed_data = data_filtered = data[data['SiteID'].isin(sites_over_500)]

# Encode categorical variables
encode = ['HostCategory', 'ResStatus', 'WinterStage']
encoder = OrdinalEncoder()
processed_data[encode] = encoder.fit_transform(data_filtered[encode])

# Sort by SiteID
processed_data = processed_data.sort_values(by='SiteID')

# Remove infinity values and large outliers in 'y'
processed_data = processed_data.replace([np.inf, -np.inf], np.nan)

processed_data.to_csv("population_growth.csv")


