<h2>Summary</h2>
<p>
  Extracting several performance metrics of an Artificial Pancreas system from sensor data.<br>
  Metrics computed for manual mode and auto mode:
</p>
<ul>
  <li>Percentage time in hyperglycemia (CGM > 180 mg/dL),</li>
  <li>Percentage of time in hyperglycemia critical (CGM > 250 mg/dL),</li>
  <li>Percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL),</li>
  <li>Percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL),</li>
  <li>Percentage time in hypoglycemia level 1 (CGM < 70 mg/dL), and</li>
  <li>Percentage time in hypoglycemia level 2 (CGM < 54 mg/dL).</li>
</ul>

<p>
  Given:
  <ul>
    <li>CGMData.csv from Continuous Glucose Sensor and InsulinData.csv from the insulin pump.</li>
    <li>Output of the CGM sensor: Data time stamp (col B and C), 5-min filtered CGM reading in mg/dL (col AE), and ISIG value/raw sensor output every 5 min.</li>
    <li>Auto Mode turned off at (row 40133, col Q), row (0 to 40133) == manual mode.</li>
    <li>Data is in reverse order of time.</li>
  </ul>
</p>

<h2>Solution</h2>
<ol>
  <li>Separate manual mode data from auto mode data using InsulinData.csv.</li>
  <li>Exclude data for days with more than 10% missing data.</li>
  <li>Fill the rest of the missing data using KNN regression and linear interpolation (288 data points per day).</li>
  <li>Classify glucose levels.</li>
  <li>Print the results to Result.csv.</li>
</ol>
