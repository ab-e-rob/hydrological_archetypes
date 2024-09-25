from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score

"""
This script performs clustering analysis on wetland data, visualizes the results, 
calculates the variance inflation factor for variable selection, and generates 
correlation matrices to understand the relationships between variables.

Key functionalities:
- Read and preprocess wetland data
- Perform K-Means clustering
- Visualize cluster centroids and wetlands within clusters
- Calculate and visualize correlation matrix with significance levels
- Evaluate multicollinearity using Variance Inflation Factor (VIF)

Data files used:
- '../results/combined_features.csv': Input data for wetland features
- Results saved to '../results/' and '../figures/'
"""


# Choose the optimal k based on elbow curve analysis
optimal_k = 6

### Read and preprocess the data
df = pd.read_csv('../results/combined_features.csv', delimiter=',', header=0)
wetland_df = pd.read_csv('../results/combined_features.csv', delimiter=',', header=0)

Dataset = True
Date = True
Mean_Area = True
Year = True
Month = True

# Drop the columns for the variables that will not used in the clustering
# True = drop column
Max_Month = True
Min_Month = True
Skewness = False
Kurtosis = False
Norm_Range = True
Std_Dev = True
Norm_Max_Slope = False
Norm_Min_Slope = True
Spr_Sum_Diff = True
Spr_Slope_Diff = True
Norm_Slope_Variation = True
CoV = True
Num_Peaks = False


# Drop specified columns based on the boolean flags
if Max_Month == True:
    df = df.drop(['Max_Month'], axis=1)
if Min_Month == True:
    df = df.drop(['Min_Month'], axis=1)
if Skewness == True:
    df = df.drop(['Skewness'], axis=1)
if Kurtosis == True:
    df = df.drop(['Kurtosis'], axis=1)
if Norm_Range == True:
    df = df.drop(['Norm_Range'], axis=1)
if Std_Dev == True:
    df = df.drop(['Std_Dev'], axis=1)
if Norm_Max_Slope == True:
    df = df.drop(['Norm_Max_Slope'], axis=1)
if Norm_Slope_Variation == True:
    df = df.drop(['Norm_Slope_Variation'], axis=1)
if Spr_Sum_Diff == True:
    df = df.drop(['Spr_Sum_Diff'], axis=1)
if Norm_Min_Slope == True:
    df = df.drop(['Norm_Min_Slope'], axis=1)
if Spr_Slope_Diff == True:
    df = df.drop(['Spr_Slope_Diff'], axis=1)
if CoV == True:
    df = df.drop(['CoV'], axis=1)
if Num_Peaks == True:
    df = df.drop(['Num_Peaks'], axis=1)

# Exclude specific wetlands from clustering
wetland_names_to_exclude = [
    'Mellerston', 'Falsterbo', 'Ottenby', 'Olands',
    'Traslovslage', 'Stigfjorden', 'Umealvens',
    'Blekinge', 'Fyllean', 'Morrumsan',
    'Nordrealvs', 'Skalderviken', 'Getteron',
    'Lundakrabukten', 'Klingavalsan', 'Gotlands',
    'Kallgate', 'Aloppkolen', 'Svenskundsviken'
]

for name in wetland_names_to_exclude:
    df = df[df.Name != name]

# Group by wetland name and calculate the mean for each feature
df = df.groupby('Name').mean()

############# Perform k-means clustering #############
def k_means():
    """
    Perform K-Means clustering on the wetland dataset.

    This function standardizes the data, performs K-Means clustering for a range
    of cluster numbers, evaluates the silhouette scores for each k, and saves the
    clustering results and visualizations to files.

    Returns:
        None
    """
    global df_scaled
    global cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6

    # Standardise the data before clustering
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Perform k-means clustering for different values of k= 2-10
    # plot the silhouette score
    silhouette = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        silhouette.append(silhouette_score(df_scaled, kmeans.labels_))

    # Plot silhouette scores
    plt.plot(range(2, 11), silhouette, marker='o')
    plt.title('Silhouette Score for K-Means Clustering')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.savefig('../results/silhouette_score.png')
    plt.show()

    # Fit K-Means with the optimal k
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
    df['Cluster'] = kmeans_optimal.fit_predict(df_scaled)

    df.to_csv('../results/cluster_analysis_results.csv', index=True)

    # Get cluster centroids
    centroids = kmeans_optimal.cluster_centers_

    # Create a DataFrame with cluster centroids and feature names
    centroids_df = pd.DataFrame(centroids, columns=df.columns[:-1])

    colors = ['#AA4499', '#DDCC77', '#7ED4FF', '#44AA99', '#117733', '#332288']

    # Plot the cluster centroids
    plt.figure(figsize=(10, 6))
    for i in range(optimal_k):
        plt.plot(centroids_df.columns, centroids[i], marker='o', label=f'Cluster {i + 1}', color=colors[i])

    plt.title('Cluster Centroids')
    plt.xlabel('Hydrological parameter')
    plt.ylabel('Centroid Value')
    plt.legend([cluster_names[i] for i in range(optimal_k)], loc='upper left')
    plt.savefig('../results/cluster_centroids.png')
    plt.show()

    # get a list of wetland names based on cluster
    cluster_1 = df[df['Cluster'] == 0].index.tolist()
    cluster_2 = df[df['Cluster'] == 1].index.tolist()
    cluster_3 = df[df['Cluster'] == 2].index.tolist()
    cluster_4 = df[df['Cluster'] == 3].index.tolist()
    cluster_5 = df[df['Cluster'] == 4].index.tolist()
    cluster_6 = df[df['Cluster'] == 5].index.tolist()

    # save mean values to csv
    df.groupby('Cluster').mean().to_csv('../results/cluster_means.csv', index=True)

    return


def cluster_plots():
    """
    Generate and save plots for each cluster.

    This function creates subplots for each wetland within each cluster and saves
    the figures to files. The wetland data is plotted against the month to show
    the area changes visually.

    Returns:
        None
    """
    for c, cluster in enumerate(range(1, 7)):
        current_cluster = globals()[f'cluster_{c + 1}']  # Fix the variable name dynamically
        plt.figure(figsize=(15, 10))

        num_rows = math.ceil(len(current_cluster) / 3)  # Adjust the number of columns as needed
        num_columns = 3

        colors = ['#AA4499', '#DDCC77', '#7ED4FF', '#44AA99', '#117733', '#332288']

        for i, wetland in enumerate(current_cluster):
            plt.subplot(num_rows, num_columns, i + 1)
            wetland_data = wetland_df[wetland_df['Name'] == wetland]
            plt.plot(wetland_data['Month'], wetland_data['Area (metres squared)'], label=wetland, color=colors[c], linewidth=2)
            plt.title(f'{wetland} ({cluster_names[cluster - 1]})', fontsize=15)
            plt.xticks(range(3, 11), ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'], fontsize=13)
            plt.ylabel('Water extent (ha)', fontsize=14)
            plt.yticks(fontsize=13)
            plt.legend('')

        # Add spacing between subplots
        plt.tight_layout(h_pad=1.0, w_pad=0.5)

        # Save figure
        plt.savefig(f'../results/basic_cluster_plots/cluster_{cluster}.png')

        # Show the plot
        plt.show()

    return


def calculate_variance_inflation_factor(X):
    """
    Calculate the Variance Inflation Factor (VIF) for each variable in the input DataFrame.

    Parameters:
        - X (pd.DataFrame): Input DataFrame containing features to evaluate for multicollinearity.

    Returns:
        - vif (pd.Series): DataFrame containing VIF values for each variable.
    """
    # drop the cluster column

    # Create an empty DataFrame to store the results
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # save result to csv
    vif.to_csv('../results/vif.csv', index=False)

    print(vif)

    return vif


def correlation_matrix():
    """
    Generate and visualize the correlation matrix for the dataset.

    This function calculates the correlation coefficients and associated p-values
    for each pair of variables in the DataFrame. It visualizes the correlation matrix
    with significance stars and saves the figure to a file.

    Returns:
        None
    """

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Calculate p-values for each pair of variables
    p_values = np.zeros_like(correlation_matrix)
    stars = np.empty_like(correlation_matrix, dtype='U3')  # 'U3' corresponds to a Unicode string of length 3
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            corr, p_value = pearsonr(df.iloc[:, i], df.iloc[:, j])
            p_values[i, j] = p_value
            p_values[j, i] = p_value

            if p_value < 0.05:
                stars[i, j] = '*'
                stars[j, i] = '*'
            else:
                stars[i, j] = ''
                stars[j, i] = ''

    # Convert the correlation matrix to string type with 4 significant figures
    correlation_matrix_str = correlation_matrix.round(4).astype(str)

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Visualize the lower triangle of the correlation matrix with stars for significance
    plt.figure(figsize=(12, 8))
    annot_labels = correlation_matrix_str + stars
    sns.heatmap(correlation_matrix, annot=annot_labels, cmap='coolwarm', fmt="", linewidths=.5, mask=mask)
    plt.legend(title='Significance', loc='upper left', labels=['*', ''])
    # set fontsize
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title('Correlation Matrix with Significance Stars', fontsize=16)

    # tight layout
    plt.tight_layout()

    # save fig
    plt.savefig('../figures/correlation_matrix.png')

    plt.show()

    return


# Define cluster names
cluster_names = {
    0: 'Autumn drying',
    1: 'Summer dry',
    2: 'Spring surging',
    3: 'Summer flooded',
    4: 'Spring flooded',
    5: 'Slow drying'
}

# Uncomment the following line to calculate the variance inflation factor
# df = df.drop(['Cluster'], axis=1)
# vif_result = calculate_variance_inflation_factor(df)

# Call functions
k_means()
cluster_plots()
# Uncomment to generate the correlation matrix
# correlation_matrix()






