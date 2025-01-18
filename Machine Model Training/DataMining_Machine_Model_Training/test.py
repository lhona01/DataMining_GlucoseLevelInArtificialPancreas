import pickle
import pandas as pd
import numpy as np

# 1. Time difference between peak after meal
def time_difference(record):
    filtered_record = record.iloc[:, 0:]
    max_indices = filtered_record.idxmax(axis=1)
    column_positions = max_indices.map(lambda col: record.columns.get_loc(col))
    result = column_positions * 5
    return pd.DataFrame(result, columns=['time difference'])

# feature 2: glucose level difference after the meal (max glucose level - min glucose level)
def glucose_level_difference(record):
    max_indices = record.max(axis=1)
    meal_indices = record.iloc[:, 0]
    result = (max_indices - meal_indices) / meal_indices
    return pd.DataFrame(result, columns=['glucose level difference'])

with open('decision_tree_model.pkl', 'rb') as file:
    tree = pickle.load(file)

df = pd.read_csv('test.csv')
df_time_difference = time_difference(df)
df_glucose_level_difference = glucose_level_difference(df)

final_data = pd.concat([df_time_difference, df_glucose_level_difference], axis=1)

prediction = tree.predict(final_data)

predictions_array = np.array(prediction)

predictions_df = pd.DataFrame(predictions_array)

predictions_df.to_csv('Result.csv', index=False)