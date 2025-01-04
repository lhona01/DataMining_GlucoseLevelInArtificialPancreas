import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# function converts string time to datetime
def convertToTime(stringTime):
    apple = datetime.strptime(stringTime, "%H:%M:%S")
    return apple.time()

def timeToSeconds(time_obj):
    seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return seconds

def timeToMinutes(time_obj):
    return time_obj.hour * 60 + time_obj.minute + time_obj.second / 60

def timeToHours(time_obj):
    return time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600

dataframe = pd.read_csv('CGMData.csv', usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
dataframe.rename(columns= {'Sensor Glucose (mg/dL)': 'Sensor Glucose'}, inplace=True)
dataframe = dataframe.reset_index(drop=True)
dataframe['Time'] = dataframe['Time'].apply(convertToTime)

##### assigning missing values via KNN regression
# rows with missing "Sensor Glucose (mg/dL"
uniqueDate = dataframe['Date'].unique()
missingSensorGlucose = dataframe[dataframe['Sensor Glucose'].isnull()]
glucoseReadingTimeDiff = 5
expectedDataPerDay = 288
#find missing time/gap in time and fill it in
for date in uniqueDate:
#for date in ['2/11/2018']:
    dailySensorGlucose = dataframe[dataframe['Date'] == date]
    # if more than 10% of the data is missing in a given day (don't include data from that day)
    if (len(dailySensorGlucose) < expectedDataPerDay - (.05 * expectedDataPerDay)
    or len(dailySensorGlucose) > expectedDataPerDay):
        dataframe = dataframe[dataframe['Date'] != date] #remove the following date from dataframe
        #print('deleted dates:', date)
        continue
    # adding missing time ex- 280/288 data is given, locate other 8 and add a row
    elif (expectedDataPerDay - (.1 * expectedDataPerDay)) < len(dailySensorGlucose) < expectedDataPerDay:
        dummyDate = datetime.strptime(date, "%m/%d/%Y") #to use timedelta()
        fiveMinute = timedelta(minutes=5, seconds=30)
        missingTime = [] #used to fill the gaps in time found for the given day to make sum data for the day = 288
        for row in dailySensorGlucose.iterrows():
            index = row[0]
            rowTime = datetime.combine(dummyDate, dailySensorGlucose['Time'][index])
            try:
                nextRowTime = datetime.combine(dummyDate, dailySensorGlucose['Time'][index + 1])
            except:
                break
            while (abs(nextRowTime - rowTime) > fiveMinute):
                missingTime.append({'Date': date, 'Time': (rowTime - timedelta(minutes=5)).time(), 
                        'Sensor Glucose': np.nan, 'Order': rowTime - nextRowTime})
                rowTime = rowTime - timedelta(minutes=5)
        if (len(missingTime) > 0):
            missingTime = pd.DataFrame(missingTime).sort_values(by='Order', ascending=False)
            #remove bottom rows of missing time
            if (expectedDataPerDay - len(dailySensorGlucose) < len(missingTime)):
                missingTime = missingTime[:(expectedDataPerDay - len(dailySensorGlucose) - len(missingTime))]
            #drop Order Column
            missingTime = missingTime.drop('Order', axis=1)
            dataframe = pd.concat([dataframe, missingTime])
    
# Reset index after concatenation
dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%m/%d/%Y')
dataframe = dataframe.sort_values(by=['Date', 'Time'], ascending=[False, False])
dataframe = dataframe.reset_index(drop=True)

# Convert 'Time' to seconds since midnight for interpolation
dataframe['Time'] = dataframe['Time'].apply(timeToHours)

# Perform linear interpolation on 'Sensor Glucose'
dataframe['Sensor Glucose'] = dataframe['Sensor Glucose'].interpolate(method='linear')

# Read Insulin Data
insulinData = pd.read_csv('InsulinData.csv', usecols=['Date', 'Time', 'Alarm'])

# Determine the threshold datetime
divideDataFrameAt = insulinData[insulinData['Alarm'] == "AUTO MODE ACTIVE PLGM OFF"].head(1)
divideDataFrameAt['Date'] = pd.to_datetime(divideDataFrameAt['Date'], format='%m/%d/%Y')
divideDataFrameAt['Time'] = divideDataFrameAt['Time'].apply(convertToTime)
divideDataFrameAt['Time'] = divideDataFrameAt['Time'].apply(timeToHours)

# Extract threshold date and time
threshold_date = divideDataFrameAt['Date'].values[0]
threshold_time = divideDataFrameAt['Time'].values[0]

# DataFrame with datetimes greater than the threshold
manualMode = dataframe[(dataframe['Date'] > threshold_date) | 
                ((dataframe['Date'] == threshold_date) & (dataframe['Time'] > threshold_time))]
# DataFrame with datetimes less than the threshold
autoMode = dataframe[(dataframe['Date'] < threshold_date) | 
                ((dataframe['Date'] == threshold_date) & (dataframe['Time'] < threshold_time))]

manualMode = manualMode[manualMode['Date'] != threshold_date] # removing the date the data was divided into auto and manual
autoMode = autoMode[autoMode['Date'] != threshold_date]

def classifySensorGlucoseLevel(data):
    uniqueDate = data['Date'].unique()
    # 0 = hyperglycimia critical, 1 = hyperglycemia, 2 = hypoglycemiaLevel_1, 3 = hypoglycemiaLevel_2, 4 = inRange, 5 = 'inRangeSecondary'
    metric = np.zeros((len(uniqueDate), 6))
    metricColumnIndex = -1
    
    for date in uniqueDate:
        metricColumnIndex += 1
        dailySensorGlucose = data[data['Date'] == date]
        for sensorGlucose in dailySensorGlucose['Sensor Glucose']:
            if sensorGlucose > 250:
                metric[metricColumnIndex][1] += 1
            elif sensorGlucose <= 250 and sensorGlucose > 180:
                metric[metricColumnIndex][0] += 1
            elif sensorGlucose < 70 and sensorGlucose >= 54:
                metric[metricColumnIndex][4] += 1
            elif sensorGlucose < 54:
                metric[metricColumnIndex][5] += 1
            else:
                if sensorGlucose >= 70 and sensorGlucose <= 180:
                    metric[metricColumnIndex][2] += 1
                    if sensorGlucose <= 150:
                        metric[metricColumnIndex][3] += 1
    
    # Convert numpy array to dataframe
    columnName = ['hyperglycimia', 'hyperglycimia critical', 'inRange', 'inRangeSecondary', 'hypoglycemiaLevel_1', 'hypoglycemiaLevel_2']
    metric_df = pd.DataFrame(metric, columns=columnName)
    
    metric_df =  metric_df / 288 # metric_df / avgNumDataPerDay
    
    metric_df = metric_df.mean() * 100
    return metric_df

# Calculate metrics for different time periods for both modes
allDayManualMetric = classifySensorGlucoseLevel(manualMode)
allDayAutoMetric = classifySensorGlucoseLevel(autoMode)

# 6am to 12am data
daytimeManualData = manualMode[manualMode['Time'] >= 6]
daytimeAutoData = autoMode[autoMode['Time'] >= 6]

daytimeManualMetric = classifySensorGlucoseLevel(daytimeManualData)
daytimeAutoMetric = classifySensorGlucoseLevel(daytimeAutoData)

# 12am to 6am data
morningManualData = manualMode[(manualMode['Time'] >= 0) & (manualMode['Time'] < 6)]
morningAutoData = autoMode[(autoMode['Time'] >= 0) & (autoMode['Time'] < 6)]

morningManualMetric = classifySensorGlucoseLevel(morningManualData)
morningAutoMetric = classifySensorGlucoseLevel(morningAutoData)

result_csv = np.empty((2,18))
    
#add morningManualMetric values into result_csv
result_csv[0, :6] = morningManualMetric.values
result_csv[0, 6:12] = daytimeManualMetric.values
result_csv[0, 12:] = allDayManualMetric.values

result_csv[1, :6] = morningAutoMetric.values
result_csv[1, 6:12] = daytimeAutoMetric.values
result_csv[1, 12:] = allDayAutoMetric.values

rows = ["Manula Mode", "Auto Mode"]

metrics = [
    '% of time in hyperglycemia',
    '% of time in  hyperglycemia critical',
    '% of time in range',
    '% of time in range secondary',
    '% of time in hypoglycemia level 1',
    '% of time in hypoglycemia level 2'
    ]

timeOfDay = [
    'daytime 6am - 12pm',
    'overnight 12pm - 6am',
    'all day 12am - 12pm'
]

multi_columns = pd.MultiIndex.from_product([metrics, timeOfDay], names=["Metric", "TimeOfDay"])

# Convert the result CSV matrix to a DataFrame with the correct columns and rows
result_df = pd.DataFrame(result_csv, index=rows, columns=multi_columns)

# Save the DataFrame to a CSV file without headers and index
result_df.to_csv('Result.csv', index=True, header=True)