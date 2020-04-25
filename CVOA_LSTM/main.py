from CVOA.CVOA import CVOA
from ETL.ETL import *
from DEEP_LEARNING.LSTM import *
import time as time
import os.path

if __name__ == '__main__':

    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "spain_energy_market_very_small.csv")
    # Load the dataset
    data, scaler = load_data(path_to_data=path, useNormalization=True)
    # Transform data to a supervised dataset
    data = data_to_supervised(data, historical_window=168, prediction_horizon=24)
    # Split the dataset
    xtrain, xtest, ytrain, ytest, xval, yval = splitData(data, historical_window=168, test_size=.3, val_size=.3)
    # Add shape to use LSTM network
    xtrain, xtest, ytrain, ytest, xval, yval = adaptShapesToLSTM(xtrain, xtest, ytrain, ytest, xval, yval)

    # Deep Learning parameters
    epochs = 10
    batch = 512         #1024 (changed also in LSTM, CVOA)
    
    # Initialize problem
    cvoa = CVOA(size_fixed_part = 3, min_size_var_part = 2, max_size_var_part = 11, fixed_part_max_values = [5, 8], var_part_max_value = 11, max_time = 20,
                xtrain = xtrain, ytrain=ytrain, xval=xval, yval=yval, pred_horizon=24, epochs=epochs, batch=batch, scaler = scaler)
    time = int(round(time.time() * 1000))
    solution = cvoa.run()
    time = int(round(time.time() * 1000)) - time

    print("Best solution: " + str(solution))
    print("Best fitness: " + str(CVOA.fitness(solution)))
    print("Execution time: " + str(CVOA.df.format((time) / 60000) + " mins"))