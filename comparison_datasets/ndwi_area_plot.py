import pandas as pd
import os
import calendar
import matplotlib.pyplot as plt

wetland_name = 'Aloppkolen'

# Locate csv file and load as a dataframe
file_path = r'../data/ndwi_' + wetland_name + '.csv'

df = pd.read_csv(file_path, delimiter=',', header=0)

# Extract year and month correctly
df[['year', 'month']] = df['Filename'].str.extract(r'_(\d{4})_(\d+)\.tif')

# Create a new column with the desired format
df['Date'] = df['month'].str.zfill(2) + '/' + df['year']

# Give the DataFrame a new name
df['Name'] = wetland_name

# Remove the old columns
df = df.drop('Filename', axis=1)
df = df.drop('year', axis=1)
df = df.drop('month', axis=1)

# Rearrange the old columns
df = df[['Name', 'Date', 'Area (metres squared)']]

# Display the DataFrame
print(df)

# now plot
# Convert 'new_column' to datetime for proper plotting
df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y')

# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['Date'], df['Area (metres squared)'], s=70)
plt.xlabel('Date')
plt.ylabel('Area (m^2)')
plt.title(f'{wetland_name} Area from NDWI (2020-2023)')

plt.savefig(r'../figs/ndwi_' + wetland_name + '.png')

plt.show()



