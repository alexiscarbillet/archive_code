# AiSight-Challenge
 
My work in available in 3 formats: python script, Jupiter notebook and pdf.

The results and the method used are presented in the html file.
Two datasets have been studied: project_pump.csv and project_fan.csv
Those csv are too big to be push in Github. I chose to put the final csv which contains the data after their processing. It also contains the machine state, result of the clutering.
Each dataset contains 5 columns: Unix time, Amount of samples, Time period (milliseconds), Sampling Rate, Sensor data (millivolt reads). 
The goals are to identify different machine states in the data set and cluster them and to come up with a classification algorithm that classifies the states into: State 1, 2, 3 etc...
10 machine learning techniques have been compared for the prediction of the state machine.
