import matlab.engine
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

### Start MATLAB engine
eng = matlab.engine.start_matlab()

### Load the Simulink model
eng.load_system('tracking3')

### Load data from Excel file
data = pd.read_excel("hourly data.xlsx")

### Split data into features and target variable
X = data[['temp', 'humidity', 'windspeed', 'cloudcover', 'uvindex', 'solarenergy']]  # Features
y = data['solarradiation']  # Target

### Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

### Train a decision tree regressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

### Make predictions
y_pred = model.predict(X_test)

### Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

### Define your condition based on multiple factors
condition = (
        (X_test['humidity'] > 40) & (X_test['humidity'] <= 70) &
        (X_test['windspeed'] > 5) & (X_test['windspeed'] <= 10) &
        (X_test['temp'] > 30) & (X_test['temp'] <= 35) &
        (X_test['cloudcover'] > 25) & (X_test['cloudcover'] <= 50) &
        (X_test['uvindex'] > 5) & (X_test['uvindex'] <= 10) &
        (X_test['solarenergy'] > 2) & (X_test['solarenergy'] <= 4)
)

### Filter test data based on the condition
filtered_X_test = X_test[condition]

### Main loop to send output from Simulink to machine learning model and back to Simulink
num_cycles = 30  # Number of simulation cycles

for cycle in range(num_cycles):
    ### Retrieve the output value from 'x1' in 'out'
    eng.set_param('tracking3', 'SimulationCommand', 'start', nargout=0)
    status = eng.get_param('tracking3', 'SimulationStatus')
    while status != 'stopped':
        status = eng.get_param('tracking3', 'SimulationStatus')

    x1_value = eng.eval('out.x1')
    first_x1_value = x1_value[cycle][0]
    print(f'Cycle {cycle + 1}: Simulink Output = {first_x1_value:.4f}')

    ### Modify the feature data to include the Simulink output
    new_data = filtered_X_test.copy()
    new_data.iloc[:,
    0] = first_x1_value  # Assume 'temp' is the first column and use it as a placeholder for 'solarradiation'

    ### Make predictions on modified test data
    filtered_y_pred = model.predict(new_data)

    ### Calculate and print the average of the predicted values
    average_prediction = round(np.mean(filtered_y_pred), 6)
    print("Average of Predicted Values:", average_prediction)

    ### Set the value for the constant block 'Constant' with the machine learning output
    constant_block_path = 'tracking3/Constant'
    eng.set_param(constant_block_path, 'Value', str(average_prediction), nargout=0)

    ### Pause for a moment before next cycle (optional)
    time.sleep(0.1)  # Adjust as needed

### Close the Simulink model without saving
eng.close_system('tracking3', 0, nargout=0)

### Stop MATLAB engine
eng.quit()
