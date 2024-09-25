import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Wetland names dictionary
wetland_names = {
    "Wetland name": [
        "Aloppkolen", "Annsjon", "Askoviken", "Asnen", "Blekinge",
        "Dattern", "Dummemosse", "Eman", "Falsterbo", "Farnebofjarden", "Fyllean",
        "Gammelstadsviken", "Getapulien", "Getteron", "Gotlands", "Gullhog",
        "Gustavmurane", "Helge", "Hjalstaviken", "Hornborgasjon", "Hovramomradet",
        "Kallgate", "Kilsviken", "Klingavalsan", "Komosse", "Koppangen", "Kvismaren",
        "Laidaure", "Lundakrabukten", "Maanavuoma", "Mellanljusnan", "Mellerston",
        "Morrumsan", "Mossatrask", "Nittalven", "Nordrealvs", "Olands", "Oldflan",
        "Oset", "Osten", "Ottenby", "Ovresulan", "Paivavouma", "Persofjarden",
        "Pirttimysvuoma", "Rappomyran", "Sikavagarna", "Skalderviken",
        "Stigfjorden", "Storemosse", "Storkolen", "Svartadalen",
        "Svenskundsviken", "Takern", "Tarnsjon", "Tavvovouma", "Tjalmejaure",
        "Tonnersjoheden", "Traslovslage", "Tysoarna", "Vasikkavouma",
        "Vattenan"
    ]
}

### Preprocess the data ########################################################

# Locate csv file with average monthly area between 2020-2023
csv = '../data/final_data.csv'

# Read the dataset
df = pd.read_csv(csv)

# Filter dataset by model type
df = df[df['Model'] == 'deepaqua']

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Extract month from the date
df['Month'] = df['Date'].dt.month

# Calculate the average monthly area between 2020 and 2023 for each wetland
df['Year'] = df['Date'].dt.year
df_monthly_avg = df.groupby(['Name', 'Month'])['Area (metres squared)'].mean().reset_index()

# Filter the dataset to include only the wetlands in the data dictionary
df_filtered = df_monthly_avg[df_monthly_avg['Name'].isin(wetland_names['Wetland name'])]

# Filter to exclude months 11, 12, 1 and 2
df_filtered = df_filtered[~df_filtered['Month'].isin([11, 12, 1, 2])]

# convert m2 to ha
df_filtered['Area (metres squared)'] = df_filtered['Area (metres squared)'] / 10000

print(df_filtered)

### Define functions ###########################################################
def max_min_months(df_input):
    """
    Determine the months with the maximum and minimum area for each wetland.

    Args:
        df_input (pd.DataFrame): Input DataFrame containing wetland data.

    Returns:
        pd.DataFrame: DataFrame containing the name of the wetland,
                      month of maximum area, and month of minimum area.
    """
    max_months = []
    min_months = []
    wetlands = df_input['Name'].unique()

    for wetland in wetlands:
        wetland_data_single = df_input[df_input['Name'] == wetland]

        if not wetland_data_single.empty:
            max_month = wetland_data_single.loc[wetland_data_single['Area (metres squared)'].idxmax()]['Month']
            min_month = wetland_data_single.loc[wetland_data_single['Area (metres squared)'].idxmin()]['Month']
        else:
            max_month = min_month = None

        max_months.append(max_month)
        min_months.append(min_month)

    df_max_min = pd.DataFrame({'Name': wetlands, 'Max_Month': max_months, 'Min_Month': min_months})

    return df_max_min


def std_dev(df_input):
    """
    Calculate the standard deviation of the area for each wetland.

    Args:
        df_input (pd.DataFrame): Input DataFrame containing wetland data.

    Returns:
        pd.DataFrame: DataFrame containing the name of the wetland and its standard deviation of area.
    """
    std_dev_list = []
    wetlands = df_input['Name'].unique()

    for wetland in wetlands:
        wetland_data_single = df_input[df_input['Name'] == wetland]

        if not wetland_data_single.empty:
            std_dev_value = wetland_data_single['Area (metres squared)'].std()
        else:
            std_dev_value = None

        std_dev_list.append(std_dev_value)

    df_std_dev = pd.DataFrame({'Name': wetlands, 'Std_Dev': std_dev_list})
    return df_std_dev

def skewness(df_input):
    """
    Calculate the skewness of the area for each wetland.

    Args:
        df_input (pd.DataFrame): Input DataFrame containing wetland data.

    Returns:
        pd.DataFrame: DataFrame containing the name of the wetland and its skewness of area.
    """
    skewness_list = []
    wetlands = df_input['Name'].unique()

    for wetland in wetlands:
        wetland_data_single = df_input[df_input['Name'] == wetland]

        if not wetland_data_single.empty:
            skewness_value = wetland_data_single['Area (metres squared)'].skew()
        else:
            skewness_value = None

        skewness_list.append(skewness_value)

    df_skewness = pd.DataFrame({'Name': wetlands, 'Skewness': skewness_list})
    return df_skewness

def kurtosis(df_input):
    """
    Calculate the kurtosis of the area for each wetland.

    Args:
        df_input (pd.DataFrame): Input DataFrame containing wetland data.

    Returns:
        pd.DataFrame: DataFrame containing the name of the wetland and its kurtosis of area.
    """
    kurtosis_list = []
    wetlands = df_input['Name'].unique()

    for wetland in wetlands:
        wetland_data_single = df_input[df_input['Name'] == wetland]

        if not wetland_data_single.empty:
            kurtosis_value = wetland_data_single['Area (metres squared)'].kurtosis()
        else:
            kurtosis_value = None

        kurtosis_list.append(kurtosis_value)

    df_kurtosis = pd.DataFrame({'Name': wetlands, 'Kurtosis': kurtosis_list})
    return df_kurtosis

def range(df_input):
    """
    Calculate the normalised range of the area for each wetland.

    Args:
        df_input (pd.DataFrame): Input DataFrame containing wetland data.

    Returns:
        pd.DataFrame: DataFrame containing the name of the wetland and its normalised range of area.
    """
    range_list = []
    wetlands = df_input['Name'].unique()

    for wetland in wetlands:
        wetland_data_single = df_input[df_input['Name'] == wetland]

        if not wetland_data_single.empty:
            max_area = wetland_data_single['Area (metres squared)'].max()
            min_area = wetland_data_single['Area (metres squared)'].min()
            range_value = (max_area - min_area) / wetland_data_single['Area (metres squared)'].mean()
        else:
            range_value = None

        range_list.append(range_value)

    df_range = pd.DataFrame({'Name': wetlands, 'Norm_Range': range_list})
    return df_range

def rate_of_change(df_input):
    """
    Calculate the normalised maximum and minimum rate of change for each wetland.

    Args:
        df_input (pd.DataFrame): Input DataFrame containing wetland data.

    Returns:
        pd.DataFrame: DataFrame containing the name of the wetland, normalised maximum slope,
                      and normalised minimum slope.
    """
    max_slope_list = []
    min_slope_list = []
    wetlands = df_input['Name'].unique()

    for wetland in wetlands:
        wetland_data_single = df_input[df_input['Name'] == wetland]

        if not wetland_data_single.empty:
            wetland_data_single['Rate_of_Change'] = wetland_data_single['Area (metres squared)'].diff()
            max_slope = wetland_data_single['Rate_of_Change'].max()
            min_slope = wetland_data_single['Rate_of_Change'].min()
            range_value = wetland_data_single['Area (metres squared)'].max() - \
                          wetland_data_single['Area (metres squared)'].min()
            normalized_max_slope = max_slope / range_value
            normalized_min_slope = min_slope / range_value
        else:
            normalized_max_slope = normalized_min_slope = None

        max_slope_list.append(normalized_max_slope)
        min_slope_list.append(normalized_min_slope)

    df_rate_of_change = pd.DataFrame({'Name': wetlands,
                                      'Norm_Max_Slope': max_slope_list,
                                      'Norm_Min_Slope': min_slope_list})
    return df_rate_of_change

def spr_sum_diff(df_input):
    """
    Calculate the difference between the average spring water extent (March, April, May) and the average summer water
    extent (June, July and August), normalised to the mean wetland area.

    Args:
        df_input (pd.DataFrame): Input DataFrame containing wetland data.

    Returns:
        pd.DataFrame: DataFrame containing the name of the wetland and normalised spring-summer difference in area.,
    """
    spr_sum_diff_list = []
    wetlands = df_input['Name'].unique()

    for wetland in wetlands:
        wetland_data_single = df_input[df_input['Name'] == wetland]

        if not wetland_data_single.empty:
            spring_data = wetland_data_single[wetland_data_single['Month'].isin([3, 4, 5])]
            summer_data = wetland_data_single[wetland_data_single['Month'].isin([6, 7, 8])]
            mean_area_spring = spring_data['Area (metres squared)'].mean()
            mean_area_summer = summer_data['Area (metres squared)'].mean()
            diff = (mean_area_summer - mean_area_spring) / wetland_data_single['Area (metres squared)'].mean()
        else:
            diff = None

        spr_sum_diff_list.append(diff)

    df_spr_sum_diff = pd.DataFrame({'Name': wetlands, 'Spr_Sum_Diff': spr_sum_diff_list})
    return df_spr_sum_diff

def spr_slope_difference(df_input):
    """
    Calculate the difference between the average spring slope (March, April, May) and the average summer slope
    (June, July and August), normalised to the mean wetland area.

    Args:
        df_input (pd.DataFrame): Input DataFrame containing wetland data.

    Returns:
        pd.DataFrame: DataFrame containing the name of the wetland and normalised spring-summer difference in slope.,
    """
    normalized_spr_slope_diff_list = []
    wetlands = df_input['Name'].unique()

    for wetland in wetlands:
        wetland_data_single = df_input[df_input['Name'] == wetland]

        if not wetland_data_single.empty:
            spring_data = wetland_data_single[wetland_data_single['Month'].isin([3, 4, 5])]
            summer_data = wetland_data_single[wetland_data_single['Month'].isin([6, 7, 8])]

            if len(spring_data) > 1 and len(summer_data) > 1:
                spring_slope = np.polyfit(spring_data.index, spring_data['Area (metres squared)'], 1)[0]
                summer_slope = np.polyfit(summer_data.index, summer_data['Area (metres squared)'], 1)[0]
                slope_diff = summer_slope - spring_slope
                normalized_spr_slope_diff = slope_diff / wetland_data_single['Area (metres squared)'].mean()
            else:
                normalized_spr_slope_diff = None
        else:
            normalized_spr_slope_diff = None

        normalized_spr_slope_diff_list.append(normalized_spr_slope_diff)

    df_spr_slope_diff = pd.DataFrame({'Name': wetlands, 'Spr_Slope_Diff': normalized_spr_slope_diff_list})
    return df_spr_slope_diff


def slope_variation(df_input):
    """
    Calculate the standard deviation of all month-to-month slopes of monthly water extent change, normalised to the
    mean wetland area

    Args:
        df_input (pd.DataFrame): Input DataFrame containing wetland data.

    Returns:
        pd.DataFrame: DataFrame containing the name of the wetland and normalised variation of month-to-month slopes
    """
    # Initialize lists to store slope variation values and wetland names
    slope_variation_list = []
    wetland_names_list = []
    wetlands = df_input['Name'].unique()

    # Calculate the slopes of the wetland area across all months
    for wetland in wetlands:
        wetland_data = df_input[df_input['Name'] == wetland]

        if len(wetland_data) <= 1:
            # If there are not enough data points, assign None to the slope variation
            normalized_slope_variation = None
        else:
            # Calculate the slopes of the wetland area over each time step
            slopes = np.gradient(wetland_data['Area (metres squared)'])

            # Calculate the standard deviation of slopes
            slope_variation_value = np.std(slopes)

            # Calculate the difference between the max and min monthly area
            max_area = wetland_data['Area (metres squared)'].max()
            min_area = wetland_data['Area (metres squared)'].min()
            range = max_area - min_area

            # Normalize the slope variation by the mean wetland size
            normalized_slope_variation = slope_variation_value / range

        # Append the normalized slope variation value to the list
        slope_variation_list.append(normalized_slope_variation)
        wetland_names_list.append(wetland)

    # Create a DataFrame with Wetland Name and Normalized Slope Variation
    df_slope_variation = pd.DataFrame({'Name': wetland_names_list, 'Norm_Slope_Variation': slope_variation_list})

    return df_slope_variation

def cov(df_input):
    """
    Calculate the dispersion of water extent values around the mean

    Args:
        df_input (pd.DataFrame): Input DataFrame containing wetland data.

    Returns:
        pd.DataFrame: DataFrame containing the name of the wetland and coefficient of variation
    """
    # Initialize lists to store coefficient of variation values and wetland names
    cov_list = []
    wetland_names_list = []
    wetlands = df_input['Name'].unique()

    # Calculate coefficient of variation
    for wetland in wetlands:
        wetland_data_single = df_input[df_input['Name'] == wetland]

        if not wetland_data_single.empty:
            # Calculate the coefficient of variation for Mean_Area
            mean_area_mean = wetland_data_single['Area (metres squared)'].mean()
            mean_area_std = wetland_data_single['Area (metres squared)'].std()

            if mean_area_mean != 0:
                cov_value = mean_area_std / mean_area_mean
            else:
                cov_value = None
        else:
            # If wetland_data is empty, assign None to the coefficient of variation
            cov_value = None

        # Append the coefficient of variation value to the list
        cov_list.append(cov_value)
        wetland_names_list.append(wetland)

    # Create a DataFrame with Wetland Name and Coefficient of Variation
    df_cov = pd.DataFrame({'Name': wetland_names_list, 'CoV': cov_list})

    return df_cov

def peaks(df_input, prominence_threshold=10):
    """
    Identify the peak months with significant area changes for each wetland.

    Args:
        df_input (pd.DataFrame): Input DataFrame containing wetland data.

    Returns:
        pd.DataFrame: DataFrame containing the name of the wetland and its peak values.
    """
    # Initialize a list to store the number of peaks for each wetland
    peaks_list = []
    wetland_names_list = []

    # Get unique wetland names
    wetlands = df_input['Name'].unique()

    # Calculate peaks for each wetland
    for wetland in wetlands:
        wetland_data = df_input[df_input['Name'] == wetland]

        if not wetland_data.empty:
            # Find peaks in the 'Area (metres squared)' column
            peaks_indices, _ = find_peaks(wetland_data['Area (metres squared)'], prominence=prominence_threshold)

            # Count the number of peaks
            num_peaks = len(peaks_indices)
        else:
            # If wetland_data is empty, assign None
            num_peaks = None

        # Append the number of peaks and the wetland name to the lists
        peaks_list.append(num_peaks)
        wetland_names_list.append(wetland)

    # Create a DataFrame with Wetland Name and Number of Peaks
    df_peaks = pd.DataFrame({'Name': wetland_names_list, 'Num_Peaks': peaks_list})

    return df_peaks

def baseline_months(df_input):
    """
    Calculate the number of months as a fraction of total months within the 25th percentile of the distribution of
    monthly water extent change, normalised to the water extent range

    Args:
        df_input (pd.DataFrame): Input DataFrame containing wetland data.

    Returns:
        pd.DataFrame: DataFrame containing the name of the wetland and the baseline month fraction
    """
    # Initialize lists to store the results
    baseline_fraction_list = []
    wetland_names_list = []

    # Get unique wetland names
    wetlands = df_input['Name'].unique()

    # Calculate the baseline months for each wetland
    for wetland in wetlands:
        wetland_data = df_input[df_input['Name'] == wetland]

        if not wetland_data.empty:
            # Calculate the difference between the max and min monthly area
            max_area = wetland_data['Area (metres squared)'].max()
            min_area = wetland_data['Area (metres squared)'].min()
            range_area = max_area - min_area

            # Calculate the 25% percentile of the range
            percentile_25 = min_area + 0.25 * range_area

            # Count the number of months below the 25% percentile
            baseline_months_count = wetland_data[wetland_data['Area (metres squared)'] < percentile_25].shape[0]

            # Calculate the fraction of baseline months
            total_months = wetland_data.shape[0]
            baseline_fraction = baseline_months_count / total_months
        else:
            # If wetland_data is empty, assign None to the fraction
            baseline_fraction = None

        # Append the fraction value and the wetland name to the lists
        baseline_fraction_list.append(baseline_fraction)
        wetland_names_list.append(wetland)

    # Create a DataFrame with Wetland Name and Baseline Fraction
    df_baseline_fraction = pd.DataFrame({'Name': wetland_names_list, 'Baseline_Fraction': baseline_fraction_list})

    return df_baseline_fraction

# Combine the features into a single dataframe
def combine_features(df_input):
    df_max_min_months = max_min_months(df_input)
    df_std_dev = std_dev(df_input)
    df_skewness = skewness(df_input)
    df_kurtosis = kurtosis(df_input)
    df_range = range(df_input)
    df_rate_of_change = rate_of_change(df_input)
    df_spr_sum_diff = spr_sum_diff(df_input)
    df_spr_slope_diff = spr_slope_difference(df_input)
    df_slope_variation = slope_variation(df_input)
    df_cov = cov(df_input)
    df_peaks = peaks(df_input)
    df_baseline_fraction = baseline_months(df_input)

    combined_df = df_max_min_months.merge(df_std_dev, on='Name')
    combined_df = combined_df.merge(df_skewness, on='Name')
    combined_df = combined_df.merge(df_kurtosis, on='Name')
    combined_df = combined_df.merge(df_range, on='Name')
    combined_df = combined_df.merge(df_rate_of_change, on='Name')
    combined_df = combined_df.merge(df_spr_sum_diff, on='Name')
    combined_df = combined_df.merge(df_spr_slope_diff, on='Name')
    combined_df = combined_df.merge(df_slope_variation, on='Name')
    combined_df = combined_df.merge(df_cov, on='Name')
    combined_df = combined_df.merge(df_peaks, on='Name')
    combined_df = combined_df.merge(df_baseline_fraction, on='Name')

    return combined_df

# Generate combined dataframe with all features
combined_features_df = combine_features(df_filtered)

# Merge back with the monthly extent data
merged_df = pd.merge(combined_features_df, df_filtered, on=['Name'], how='left')

# Save the combined features dataframe to a CSV file
merged_df.to_csv('../results/combined_features.csv', index=False)
