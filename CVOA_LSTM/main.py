from CVOA.CVOA import CVOA
from ETL.ETL import *
from DEEP_LEARNING.LSTM import *
import time as t
import os
import sys

if __name__ == '__main__':
    

    # Deep Learning parameters
    epochs = 10
    batch = 512         #1024 (changed also in LSTM, CVOA)

    if sys.argv[1] == "t":
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "data\\UK_Test.csv")
        # Load the dataset
        data, scaler = load_data(path_to_data=path, useNormalization=True)
        # Transform data to a supervised dataset
        data = data_to_supervised(data, historical_window=9, prediction_horizon=1)
        X = data.iloc[:, 0: 9]
        Y = data.iloc[:, 9:]
        xtest = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
        ytest = np.reshape(Y.values, (Y.shape[0], Y.shape[1], 1))

        model_path = sys.argv[2]
        model = keras.models.load_model(model_path)
        model.compile(loss='mape', optimizer='adam', metrics=['mse', 'mae', 'mape'])

        predictions = model.predict(xtest[:])
        pred = scaler.inverse_transform(predictions.reshape(1, -1)).flatten()

        results = model.evaluate(xtest, ytest)
        print(predictions, pred)
        print(dict(zip(model.metrics_names, results)))
    elif sys.argv[1] == "tt":

        my_path = os.path.abspath(os.path.dirname(__file__))
        for dataset_path in os.listdir(os.path.join(my_path, "data")):
            if not dataset_path.endswith("Test.csv"):
                continue

            path = os.path.join(my_path, "data", dataset_path)
            
            # Load the dataset
            data, scaler = load_data(path_to_data=path, useNormalization=True)
            # Transform data to a supervised dataset
            data = data_to_supervised(data, historical_window=9, prediction_horizon=1)
            X = data.iloc[:, 0: 9]
            Y = data.iloc[:, 9:]
            xtest = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
            ytest = np.reshape(Y.values, (Y.shape[0], Y.shape[1], 1))

            model_path = sys.argv[2]
            model = keras.models.load_model(model_path)
            model.compile(loss='mape', optimizer='adam', metrics=['mse', 'mae', 'mape'])

            predictions = model.predict(xtest[:])
            pred = scaler.inverse_transform(predictions.reshape(1, -1)).flatten()

            results = model.evaluate(xtest, ytest)
            print(predictions, pred, dataset_path)
            print(dict(zip(model.metrics_names, results)))

    else:
    
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "data\\US_-10.csv")
        # Load the dataset
        data, scaler = load_data(path_to_data=path, useNormalization=True)
        # Transform data to a supervised dataset
        data = data_to_supervised(data, historical_window=9, prediction_horizon=1)
        # Split the dataset
        xtrain, xtest, ytrain, ytest, xval, yval = splitData(data, historical_window=9, test_size=.1, val_size=.3)
        # Add shape to use LSTM network
        xtrain, xtest, ytrain, ytest, xval, yval = adaptShapesToLSTM(xtrain, xtest, ytrain, ytest, xval, yval)

        # Initialize problem
        cvoa = CVOA(size_fixed_part=3, min_size_var_part=2, max_size_var_part=11, fixed_part_max_values=[5, 8], var_part_max_value=11, max_time=20,
                    xtrain=xtrain, ytrain=ytrain, xval=xval, yval=yval, pred_horizon=1, epochs=epochs, batch=batch, scaler=scaler)
        time = int(round(t.time() * 1000))
        solution = cvoa.run()
        time = int(round(t.time() * 1000)) - time

        print("********************************************")
        print("Best solution: " + str(solution))
        print("Best fitness: ", "{:.4f}".format(solution.fitness))
        print("Execution time: " + str((time) / 60000) + " mins")