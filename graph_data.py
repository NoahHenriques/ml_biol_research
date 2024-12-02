import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np

data = pd.read_csv("population_growth.csv")
X = data['CommonName']
y = data['GrowthRate']

data.boxplot(column='GrowthRate', by='CommonName', grid=False, vert=False)

# Set title and labels for the plot
plt.title('Growth Rate Distribution by Species (Common Name)')
plt.suptitle('')  # Suppress the default title
plt.xlabel('Growth Rate')
plt.ylabel('Species (Common Name)')

# Display the plot
#plt.show()

# Count the unique species-year pairs for each site
unique_species_year_pairs_per_site = data.groupby('SiteID').apply(lambda x: x[['CommonName', 'Year']].drop_duplicates().shape[0]).reset_index()
unique_species_year_pairs_per_site.columns = ['SiteID', 'UniqueSpeciesYearPairs']

# Count the number of sites with more than 500 unique species-year pairs
sites_over_500 = unique_species_year_pairs_per_site[unique_species_year_pairs_per_site['UniqueSpeciesYearPairs'] > 500].shape[0]

print("Number of sites with more than 500 unique species-year pairs:", sites_over_500)
