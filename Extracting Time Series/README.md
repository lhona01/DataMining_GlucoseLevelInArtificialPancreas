<h2>Summary</h2>
Extracting several performance metrics of an Aritificial Pancreas system from sensor data.
<br>
Metrics computed for manual mode and auto mode:
  1. Percentage time in hyperglycemia (CGM > 180 mg/dL),
  2. percentage of time in hyperglycemia critical (CGM > 250 mg/dL),
  3. percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL),
  4. percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL),
  5. percentage time in hypoglycemia level 1 (CGM < 70 mg/dL), and
  6. percentage time in hypoglycemia level 2 (CGM < 54 mg/dL).

Given:
  - CGMData.csv from Continous Glucose Sensor and InsulinData.csv from the insulin pump.
  - output of the CGM sensor: Data time stamp (col B and C), 5-min filtered CGM reding in mg/dL (col AE), and ISIG value/raw sensor output every 5 min.
  - Auto Mode turned off at (row 40133, col Q), row (0 to 40133) == manual mode.
  - Data is in reverse order of time.

<h2>Solution</h2>

  1. Seperate manual mode data from auto mode data using InsulinData.csv
  2. exclude data for day with more than 10% missing data.
  3. Fill rest of the missing data using KNN regression and linear interpolation (288 data per day)
  4. Classify glucose level
  5. print it to Result.csv
