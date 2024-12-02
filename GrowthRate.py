import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded CSV file
file_path = 'population_growth.csv'
population_growth_df = pd.read_csv(file_path)

def plot_species_growth(site_id, df, y_min=-10, y_max=100):

    # Filter data for the given SiteID
    site_data = df[df['SiteID'] == site_id]

    # Ensure there is data for the site
    if site_data.empty:
        print(f"No data available for SiteID {site_id}.")
        return

    # Plot each species' growth rate over time
    for species in site_data['CommonName'].unique():
        species_data = site_data[site_data['CommonName'] == species]
        plt.plot(species_data['Year'], species_data['GrowthRate'], marker='o', label=species)

    # Customize the plot
    plt.title(f"Species Growth Rate for Site {site_id}", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Growth Rate", fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Common Name")
    plt.grid(True)
    plt.ylim(y_min, y_max)

    # Show the plot
    plt.show()


plot_species_growth(48, population_growth_df,y_max=40)