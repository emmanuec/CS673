import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataSet(object):

    def __init__(self, input_train, output_train, input_test, output_test, input_validation, output_validation):
        self.input_train = input_train
        self.output_train = output_train
        self.input_test = input_test
        self.output_test = output_test
        self.input_validation = input_validation
        self.output_validation = output_validation

    @property
    def input_train(self):
        return self.__input_train

    @input_train.setter
    def input_train(self, input_train):
        self.__input_train = input_train

    @property
    def output_train(self):
        return self.__output_train

    @output_train.setter
    def output_train(self, output_train):
        self.__output_train = output_train

    @property
    def input_test(self):
        return self.__input_test

    @input_test.setter
    def input_test(self, input_test):
        self.__input_test = input_test

    @property
    def output_test(self):
        return self.__output_test

    @output_test.setter
    def output_test(self, output_test):
        self.__output_test = output_test

    @property
    def input_validation(self):
        return self.__input_validation

    @input_validation.setter
    def input_validation(self, input_validation):
        self.__input_validation = input_validation

    @property
    def output_validation(self):
        return self.__output_validation

    @output_validation.setter
    def output_validation(self, output_validation):
        self.__output_validation = output_validation


def get_data(file_name):
    # Import util
    data = pd.read_csv("D:\\Files\\Box Sync\\classes\\Spring2018\\CS673\\final\\data\\" + file_name)

    add_stock_features(data)
    data = remove_useless_data(data)

    # Dimensions of dataset
    n = data.shape[0]
    p = data.shape[1]
    # Make util a numpy array
    data = data.values

    # Training and test util
    train_start = 0
    train_end = int(np.floor(0.8 * n))
    test_start = train_end
    test_end = n
    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]

    # Scale util
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # Build DataSet
    x_train = data_train[:len(data_train) - 1, :]
    y_train = data_train[1:len(data_train), 3]
    x_test = data_test[:len(data_test) - 1, :]
    y_test = data_test[1:len(data_test), 3]

    return DataSet(x_train, y_train, x_test, y_test, None, None)


def add_stock_features(stock_data):
    stock_data["Short Moving Avg"] = stock_data["Close"].rolling(window=20).mean()
    stock_data["Long Moving Avg"] = stock_data["Close"].rolling(window=100).mean()
    stock_data["EWMA Short"] = stock_data["Close"].ewm(span=20, adjust=False).mean()
    stock_data["EWMA Long"] = stock_data["Close"].ewm(span=100, adjust=False).mean()
    stock_data["Moving Average Deviation Rate"] = stock_data["Close"].rolling(window=60).std()
    stock_data["MACD"] = (stock_data["EWMA Short"] - stock_data["EWMA Long"])
    stock_data["RSI"] = calc_rsi(stock_data["Close"], 20)


def remove_useless_data(stock_data):
    stock_data = stock_data.drop(['Date'], 1)
    stock_data = stock_data.drop(['Ex-Dividend'], 1)
    stock_data = stock_data.drop(['Split Ratio'], 1)
    stock_data.dropna(inplace=True)

    return stock_data


def calc_rsi(series, period):
        delta = series.diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        u[u.index[period - 1]] = np.mean(u[:period])  # first value is sum of avg gains
        u = u.drop(u.index[:(period - 1)])
        d[d.index[period - 1]] = np.mean(d[:period])  # first value is sum of avg losses
        d = d.drop(d.index[:(period - 1)])
        rs = pd.stats.moments.ewma(u, com=period - 1, adjust=False) / \
             pd.stats.moments.ewma(d, com=period - 1, adjust=False)
        return 100 - 100 / (1 + rs)


def get_time_data(n_history, file_name):
    # Import util
    data = pd.read_csv("D:\\Files\\Box Sync\\classes\\Spring2018\\CS673\\final\\data\\" + file_name)

    add_stock_features(data)
    data = remove_useless_data(data)

    # Dimensions of dataset
    n = data.shape[0]
    p = data.shape[1]
    # Make util a numpy array
    data = data.values

    # Training and test util
    train_start = 0
    train_end = int(np.floor(0.8 * n))
    test_start = train_end
    test_end = n
    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]

    # Scale util
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # Build DataSet
    x_train = []
    for x in range(len(data_train) - n_history):
        x_train.append(data_train[x:x + n_history, 0:])
    y_train = data_train[n_history:, 0]
    x_test = []
    for x in range(len(data_test) - n_history):
        x_test.append(data_test[x:x + n_history, 0:])
    y_test = data_test[n_history:, 0]

    return DataSet(x_train, y_train, x_test, y_test, None, None)
