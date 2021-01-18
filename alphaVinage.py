import requests
import pandas as pd
import fbprophet
import numpy as np
import json

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_validate as cross_validation
from sklearn.model_selection import train_test_split


class Apha:

    def __init__(self, ticker: str, apikey='demo'):
        """

        :param ticker: symbol of stock
        :param apikey: api for /www.alphavantage.co
        """

        ticker = ticker.upper()
        self.symbol = ticker

        # Trying get historical data for ticker
        try:
            data = requests.get('https://www.alphavantage.co/query?'
                                'function=TIME_SERIES_DAILY_ADJUSTED&'
                                'symbol={}&'
                                'outputsize=full&'
                                'apikey={}'.format(self.symbol, apikey)).json()

        except Exception as e:
            print("Error Retrieving Data.")
            print(e)
            return

        # Create DataFrame form dict with transported matrix
        stock = pd.DataFrame.from_dict(data['Time Series (Daily)']).T

        # rename, add, remove columns
        if "Adj. Close" not in stock.columns:
            stock["Adj. Close"] = stock["5. adjusted close"]
            stock["Adj. Open"] = stock["1. open"]

        stock.drop(['1. open', '4. close', '5. adjusted close', '6. volume', '7. dividend amount', '8. split coefficient'],
                   axis='columns', inplace=True)
        stock['Date'] = stock.index
        stock.rename(columns={'2. high': 'High', '3. low': 'Low'}, inplace=True)

        # format columns in lines
        cols = stock.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        stock = stock[cols]

        # reset index
        stock.index = [x+1 for x in range(len(stock.index))]

        stock_real = stock
        self.stock_real = stock_real.copy()

        stock = stock[stock['Date'].astype('datetime64[ns]') < pd.Timestamp('2018-03-07')]

        # Columns required for prophet
        stock["ds"] = stock["Date"]
        stock["y"] = stock["Adj. Close"]
        # creade daily change volumes of ticker
        stock["Daily Change"] = stock["Adj. Close"].astype('float64') - stock["Adj. Open"].astype('float64')

        self.stock = stock.copy()

        self.min_date = min(stock["Date"])
        self.max_date = max(stock["Date"])

        self.training_years = 2

        # Prophet parameters
        # Default prior from library
        self.changepoint_prior_scale = 0.05
        self.weekly_seasonality = False
        self.daily_seasonality = True
        self.monthly_seasonality = True
        self.yearly_seasonality = True
        self.changepoints = None

        print(
            "{} Stocker Initialized. Data covers {} to {}.".format(
                self.symbol, self.min_date, self.max_date
            )
        )

    # Remove weekends from a dataframe
    def remove_weekends(self, dataframe):

        # Reset index to use ix
        dataframe = dataframe.reset_index(drop=True)

        weekends = []

        # Find all of the weekends
        for i, date in enumerate(dataframe["ds"]):
            if (date.weekday()) == 5 | (date.weekday() == 6):
                weekends.append(i)

        # Drop the weekends
        dataframe = dataframe.drop(weekends, axis=0)

        return dataframe

    # Create a prophet model without training
    def create_model(self):

        # Make the model
        model = fbprophet.Prophet(
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            changepoints=self.changepoints,
        )

        if self.monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        return model

    # Predict the future price for a given range of days
    def predict_future(self, days=30):

        start_date = pd.Timestamp('2016-10-10')
        end_date = pd.Timestamp('2018-10-10')

        # Use past self.training_years years for training
        train = self.stock[
            self.stock["Date"].astype('datetime64[ns]') >
            (max(self.stock["Date"].astype('datetime64[ns]')) - pd.DateOffset(years=self.training_years))
            ]
        # train = self.stock[
        #     start_date < self.stock[self.stock["Date"]].astype('datetime64[ns]') < end_date
        #                             ]

        model = self.create_model()

        model.fit(train)

        # Future dataframe with specified number of days to predict
        future = model.make_future_dataframe(periods=days, freq="D")
        future = model.predict(future)

        # Only concerned with future dates
        future = future[future["ds"] >= max(self.stock["Date"])]

        # Remove the weekends
        future = self.remove_weekends(future)

        # Calculate whether increase or not
        future["diff"] = future["yhat"].diff()

        future = future.dropna()

        # Find the prediction direction and create separate dataframes
        future["direction"] = (future["diff"] > 0) * 1

        # Rename the columns for presentation
        future = future.rename(
            columns={
                "ds": "Date",
                "yhat": "estimate",
                "diff": "change",
                "yhat_upper": "upper",
                "yhat_lower": "lower",
            }
        )

        future_increase = future[future["direction"] == 1]
        future_decrease = future[future["direction"] == 0]

        # Print out the dates
        print("\nPredicted Increase: \n")
        print(future_increase[["Date", "estimate", "change", "upper", "lower"]])

        print("\nPredicted Decrease: \n")
        print(future_decrease[["Date", "estimate", "change", "upper", "lower"]])

        # future.to_csv('stock-{} in {} days.csv'.format(self.symbol, days), sep='\t', encoding='utf-8',
        #               columns=["Date", "estimate", "change", "upper", "lower"])

        return future[future.estimate == future.estimate.max()].to_dict()

    def alg_reg_pred(self, days=5):

        df = self.stock
        # df['prediction'] = df['Adj. Close'].shift(-1)
        # df.dropna(inplace=True)
        # forecast_time = int(days)
        #
        # X = np.array(df['prediction']).reshape(-1, 1)
        # Y = np.array(df['prediction'])
        # X = preprocessing.scale(X)
        # # X_prediction = X[-forecast_time:]
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.55)
        #
        # # Performing the Regression on the training data
        # clf = LinearRegression()
        # clf.fit(X_train, Y_train)
        # prediction = (clf.predict(X_prediction))
        #
        # last_row = df.tail(1)
        # print(last_row['Date'], last_row['Adj. Close'])
        #
        # if (float(prediction[4]) > (float(last_row['Adj. Close'])) + 1):
        #     output = (
        #             "\n\nStock:" + str(self.symbol) +
        #             "\nPrior Close:\n" + str(last_row['Adj. Close']) +
        #             "\n\nPrediction in 1 Day: " + str(prediction[0]) +
        #             "\nPrediction in 5 Days: " + str(prediction[4])
        #     )
        #     print(output)

        forecast_time = int(days)
        df['prediction'] = df['Adj. Close'].shift(-1)
        x = np.array(df['prediction']).reshape(-1, 1)
        y = df.loc[:, 'Adj. Close']
        X_prediction = x[-forecast_time:]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
        model = LinearRegression().fit(x_train, y_train)
        y_pred = model.predict(X_prediction)
        print(y_pred)





