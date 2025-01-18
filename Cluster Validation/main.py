from matplotlib import pyplot as plt
import pandas as pd
import math
from sklearn.cluster import DBSCAN, KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import KNNImputer
import numpy as np
#from scipy.stats import entropy
from scipy.stats import mode

# 1. Time difference between peak after meal
def time_difference(record, start_index):
    filtered_record = record.iloc[:, start_index:]
    max_indices = filtered_record.idxmax(axis=1)
    column_positions = max_indices.map(lambda col: record.columns.get_loc(col))
    result = (column_positions - start_index) * 5
    return pd.DataFrame(result, columns=['time difference'])

# feature 2: glucose level difference after the meal (max glucose level - min glucose level)
def glucose_level_difference(record, start_index):
    max_indices = record.max(axis=1)
    meal_indices = record.iloc[:, start_index]
    result = (max_indices - meal_indices) / meal_indices
    return pd.DataFrame(result, columns=['glucose level difference'])

# 2. perform KNN to predict missing values
def knn_imputer(record):
    imputer = KNNImputer(n_neighbors=2)

    return pd.DataFrame(imputer.fit_transform(record), columns=record.columns)

# 1. Get rid of rows thats mising more that 5% of the data
def rid_rows_threshold(record):
    threshold = 0.05 # 5%
    empty_cell_count = record.isna().sum(axis=1)
    total_cells = record.shape[1]
    rows_to_keep = empty_cell_count / total_cells < threshold
    return record[rows_to_keep]

def getTrainingData(insulin_csv, cgm_csv):
    insulin_columns_to_read = ['Date', 'Time', 'BWZ Carb Input (grams)']
    df_insulin = pd.read_csv(insulin_csv, usecols=insulin_columns_to_read)
    df_insulin['Date'] = pd.to_datetime(df_insulin['Date'], errors='coerce').dt.date
    df_insulin['Date'] = df_insulin['Date'].astype(str)
    df_insulin['datetime'] = pd.to_datetime(df_insulin['Date'] + ' ' + df_insulin['Time'], errors='coerce')
    df_insulin = df_insulin.drop(columns=['Date', 'Time'])
    df_insulin = df_insulin.rename(columns={'BWZ Carb Input (grams)': 'meal'})
    df_insulin = df_insulin.dropna(subset=['meal'])
    df_insulin = df_insulin[df_insulin['meal'] != 0]
    df_insulin = df_insulin.sort_values(by='datetime').reset_index(drop=True)

    del insulin_columns_to_read

    meal_time = []
    carb = []
    # Loop through the DataFrame to find meal times with a > 2-hour gap
    for i in range(len(df_insulin) - 1):
        current_time = df_insulin['datetime'].iloc[i]
        next_time = df_insulin['datetime'].iloc[i + 1]

        if current_time + pd.Timedelta(hours=2) < next_time:
            meal_time.append(current_time)
            carb.append(df_insulin['meal'].iloc[i])


    cgm_columns_to_read = ['Date', 'Time', 'Sensor Glucose (mg/dL)']
    df_cgm = pd.read_csv(cgm_csv, usecols=cgm_columns_to_read)
    df_cgm = df_cgm.rename(columns={'Sensor Glucose (mg/dL)': 'glucose'})
    df_cgm['datetime'] = pd.to_datetime(df_cgm['Date'] + ' ' + df_cgm['Time'])
    df_cgm = df_cgm.drop(columns=['Date', 'Time'])
    df_cgm = df_cgm.sort_values(by='datetime').reset_index(drop=True)

    del cgm_columns_to_read

    # find datetime for Sugar Glucose Level based on meal time, ex(meal time = 9:00, SugarGlucose measured after meal = 9:03), make meal time = 9:03
    index = 0
    new_meal_time = []

    for i in range(len(df_cgm)):
        if (df_cgm['datetime'].iloc[i] > meal_time[index]):
            if (df_cgm['datetime'].iloc[i] < (meal_time[index] + pd.Timedelta(minutes=5))): # Edge case: meal time and glucose measure not with in 5 min
                new_meal_time.append(df_cgm['datetime'].iloc[i])
            else:
                if (len(new_meal_time) != len(carb)):
                    del carb[len(new_meal_time)]

            if (index < len(meal_time) - 1):
                index += 1

    meal_time = new_meal_time

    del new_meal_time
    del index

    # locate meal_time and add glucose level 30 min before and 2hr after meal_time, 30 data per row including meal_time glucose level
    total_glucose_data = 30 # 2hrs:30min / 5min
    num_glucose_before_meal = 6 # 30min / 5min

    meal_data = []
    for datetime in  meal_time:
        data = []
        index = df_cgm[df_cgm['datetime'] == datetime].index[0] - num_glucose_before_meal

        for row in range(index, index + total_glucose_data + 1):
            if (row < index + total_glucose_data):
                data.append(df_cgm.loc[row, 'glucose'])

        meal_data.append(data)

    del index
    del data

    column_names = [f'col{i+1}' for i in range(len(meal_data[0]))]
    meal_data = pd.DataFrame(meal_data, columns=column_names)
    #-----------------------------Meal data (ready)------------------------------------------------------
    return meal_data, carb


meal_data, carb = getTrainingData('InsulinData.csv', 'CGMData.csv')
meal_data['carb'] = carb
meal_data.reset_index(drop=True, inplace=True)

# Handling Missing data
meal_data = rid_rows_threshold(meal_data)

# performing knn for 1st column is a bad idea because knn = 2
meal_data = meal_data.dropna(subset=['col1'])

#key to map points on bins and clusters
key = []
for i in range(len(meal_data)):
    key.append(i)

del carb
carb = pd.DataFrame()
carb['carb'] = meal_data['carb']
carb['key'] = key
meal_data.drop(columns=['carb'], inplace=True)

meal_data = knn_imputer(meal_data.iloc[:, 0:30])

########################### Feature Extraction #####################################
time_difference_meal = time_difference(meal_data, 5)

# Decision tree (Feature 1: time difference)
time_difference_meal['target'] = 1

time_difference_data = time_difference_meal

# feature 2: glucose level difference
glucose_level_difference_meal = glucose_level_difference(meal_data, 5)

glucose_level_difference_meal['target'] = 1
glucose_level_difference_data = glucose_level_difference_meal

row_means = meal_data.mean(axis=1)
row_means = pd.DataFrame(row_means, columns=['mean'])

row_max = meal_data.max(axis=1)
row_max = pd.DataFrame(row_max, columns=['max'])

final_data = pd.concat([time_difference_data, glucose_level_difference_data, row_means, row_max], axis=1)
final_data.drop(columns='target', inplace=True)
feature_len = 4
final_data['key'] = key

#Step 1: Ground truth
class Bin: ##Discretize each rows/meal data, ex if meal data range from 0 - 20 gram of carb put it in bin 1, 21 - 40 in bin 2 and so on
    def __init__(self, id, min_range, max_range):
        self.id = id
        self.min_range = min_range
        self.max_range = max_range
        self.key = []

def create_bins(carb):
    num_of_bins = math.ceil((max(carb['carb']) - min(carb['carb'])) / 20)
    bins = []

    # Initialize bins
    for i in range(num_of_bins):
        if i == 0:
            min_range = 0
            max_range = 20
        else:
            min_range = (20 * i) + 1
            max_range = 20 * (i + 1)
    
        bins.append(Bin(i, min_range, max_range))
        
    return bins

bins = create_bins(carb)

for i in range(len(carb)):
    if (carb['carb'].iloc[i] % 20) == 0:
        bin_index = math.floor(carb['carb'].iloc[i] / 20) - 1
    else:
        bin_index = math.floor(carb['carb'].iloc[i] / 20)

    bins[bin_index].key.append(carb['key'].iloc[i])

del carb

####### K-Means Clustering ########################
clustering_data = final_data.iloc[:, 0:feature_len]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(clustering_data)

# Initialize KMeans
kmeans = KMeans(n_clusters=len(bins), init='k-means++', n_init=10, random_state=0)

# Fit the model
kmeans.fit(x_scaled)

# Get the cluster labels
final_data['Cluster'] = kmeans.labels_

def SSE(data):
    sse_of_clusters = []
    distinct_clusters = sorted(data['Cluster'].unique().tolist())
    for i in distinct_clusters:
        # find centroid of each cluster
        temp_df = data[data['Cluster'] == i]
        centroid = temp_df.mean().tolist()
        temp_df = temp_df.values.tolist()

        # calculate distance from mean to all other points
        sum_cluster_all_points = 0
        for j in range(len(temp_df)):
            sum_distance = 0
            for k in range(len(centroid) - 2):
                sum_distance += math.pow(centroid[k] - temp_df[j][k], 2)
            sum_cluster_all_points += math.sqrt(sum_distance)
        sse_of_clusters.append(sum_cluster_all_points)
    return sse_of_clusters

k_mean_sse = SSE(final_data)
k_mean_sse = sum(k_mean_sse)
print("Sum of Squared Errors (SSE) K-means:", k_mean_sse)

#rows = cluster, columns = bins
k_mean_matrix = [[0 for _ in range(len(bins))] for _ in range(len(bins))]

for index, cluster in final_data.iterrows():
    for bin in bins:
        if cluster['key'] in bin.key:
            cluster_index = int(cluster['Cluster'])  # Convert to integer
            k_mean_matrix[cluster_index][bin.id] += 1
            break

def calculate_entropy(matrix):
    p_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    entropy= np.zeros(len(bins))

    # get the probabilities for each cell
    for row in range(matrix.shape[0]):  # Iterate over the number of rows
        sum = np.sum(matrix[row, :])
        for col in range(matrix.shape[1]):  # Iterate over the number of columns
            p_matrix[row][col] = matrix[row][col] / sum

    for row in range(p_matrix.shape[0]):  # Iterate over the number of rows
        for col in range(p_matrix.shape[1]):  # Iterate over the number of columns
            if p_matrix[row][col] != 0:
                entropy[row] += p_matrix[row, col] * math.log2(p_matrix[row, col])/2
        entropy[row] = -1 * entropy[row]
    
    total_entropy = 0

    for row in range(len(entropy)): #calculate total entropy
        num_elements_in_cluster = np.sum(matrix[row, :])
        total_entropy += num_elements_in_cluster * entropy[row]
    
    total_entropy = total_entropy / np.sum(matrix)
    return total_entropy

def calculate_purity(matrix):
    p_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    purity= np.zeros(len(bins))

    # get the probabilities for each cell
    for row in range(matrix.shape[0]):  # Iterate over the number of rows
        sum = np.sum(matrix[row, :])
        for col in range(matrix.shape[1]):  # Iterate over the number of columns
            p_matrix[row][col] = (2*matrix[row][col]) / sum

    for row in range(p_matrix.shape[0]):
        purity[row] = np.max(p_matrix[row])
    
    total_purity = 0

    for row in range(len(purity)): #calculate total entropy
        num_elements_in_cluster = np.sum(matrix[row, :])
        total_purity += num_elements_in_cluster * purity[row]
    
    total_purity = total_purity / np.sum(matrix)
    return total_purity

cluster_matrix = np.array(k_mean_matrix)  # Rows: clusters, Columns: classes
k_mean_entropy = calculate_entropy(cluster_matrix)
k_mean_purity = calculate_purity(cluster_matrix)

print("entropy:", k_mean_entropy)
print("k_meanss purity:", k_mean_purity)

final_data.drop(columns=['Cluster'], inplace=True)

############# DBSCAN ###################
clustering_data = final_data.iloc[:, 0:feature_len]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clustering_data)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.4, min_samples=7, metric="euclidean")
labels = dbscan.fit_predict(X_scaled)

final_data['Cluster'] = dbscan.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(f'Number of clusters: {n_clusters}')
dbscan_sse = SSE(final_data)
dbscan_sse = sum(dbscan_sse)
print(dbscan_sse)

dbscan_matrix = [[0 for _ in range(len(bins))] for _ in range(len(bins))]

for index, cluster in final_data.iterrows():
    for bin in bins:
        if cluster['key'] in bin.key:
            cluster_index = int(cluster['Cluster'])  # Convert to integer
            dbscan_matrix[cluster_index][bin.id] += 1
            break

cluster_matrix = np.array(dbscan_matrix)  # Rows: clusters, Columns: classes
dbscan_entropy = calculate_entropy(cluster_matrix)
dbscan_purity = calculate_purity(cluster_matrix)
#print("entropy:", dbscan_entropy)
#print("dbscan purity:", dbscan_purity)

result = [k_mean_sse, dbscan_sse, k_mean_entropy, dbscan_entropy, k_mean_purity, dbscan_purity]

# Convert the list to a DataFrame
df = pd.DataFrame([result])

# Save to CSV
df.to_csv('Result.csv', index=False, header=False)