import configparser
import json
import os
from datetime import datetime
import pandas as pd
from joblib import dump, load
from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from oandapyV20.endpoints.pricing import PricingStream
from oandapyV20.exceptions import V20Error
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#Load a config file containing account details or settings
config = configparser.ConfigParser()
file = os.path.join(os.getcwd(), 'config.cfg')
config.read(file)
accountID = config['OANDA']['accountID']
access_token = config['OANDA']['access_token']

client = API(access_token=access_token)
api = API(access_token=access_token, environment="practice")


def load_model():
    try:
        model = load("forex_analyzer.fa")
    except:
        print("Creating Empty Model")
        model = svm.SVC(kernel='linear', gamma='auto', C=2)
        dump(model, "forex_analyzer.fa")

    return model

def save_model(model):
    print("Saving Model")
    dump(model, "forex_analyzer.fa")



def callprint(message, tzone='US/Central'):
    from pytz import timezone
    sMessage = message
    now = datetime.now(timezone(tzone))
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return print(dt_string + " - " + sMessage)

def get_historical_data():
    instrument, granularity = "EUR_USD", "M15"
    _from = "2017-01-01T00:00:00Z"
    #_from = "{}-{}-{}T00:00:00Z".format(from_year, from_month, from_day)
    params = {
        "from": _from,
        "granularity": granularity,
        "count": 500,
    }

    for r in InstrumentsCandlesFactory(instrument=instrument, params=params):
        client.request(r)
        string_data = json.dumps(r.response.get('candles'), indent=2)

        r_data = r.response.get('candles')
        master_dataframe = pd.DataFrame()
        for dataset in tqdm(r_data):
            df = pd.DataFrame().from_dict(dataset).reset_index(drop=True)
            df['Open'] = float(dataset['mid']['o'])
            df['High'] = float(dataset['mid']['h'])
            df['Low'] = float(dataset['mid']['l'])
            df['Close'] = float(dataset['mid']['c'])
            df['Date'] = pd.to_datetime(df['time'])
            df['Date'] = df['Date'].dt.strftime('%d.%m.%Y')
            df['year'] = pd.DatetimeIndex(df['Date']).year
            df['month'] = pd.DatetimeIndex(df['Date']).month
            df['day'] = pd.DatetimeIndex(df['Date']).day
            df['dayofyear'] = pd.DatetimeIndex(df['Date']).dayofyear
            df['weekofyear'] = pd.DatetimeIndex(df['Date']).week
            df['weekday'] = pd.DatetimeIndex(df['Date']).weekday
            df['quarter'] = pd.DatetimeIndex(df['Date']).quarter
            df['is_month_start'] = (pd.DatetimeIndex(df['Date']).is_month_start)
            df['is_month_end'] = (pd.DatetimeIndex(df['Date']).is_month_end)
            master_dataframe = master_dataframe.append(df).reset_index(drop=True)
        #master_dataframe['complete'] = master_dataframe['complete'].astype(int)
        master_dataframe['is_month_start'] = master_dataframe['is_month_start'].astype(int)
        master_dataframe['is_month_end'] = master_dataframe['is_month_end'].astype(int)
        master_dataframe['mid'] = master_dataframe['mid'].astype(float)
        master_dataframe.drop(['time', 'Date'], axis=1, inplace=True)
        lab_enc = preprocessing.LabelEncoder()
        master_dataframe['mid'] = lab_enc.fit_transform(master_dataframe['mid'])
        master_dataframe['Open'] = lab_enc.fit_transform(master_dataframe['mid'])
        master_dataframe['High'] = lab_enc.fit_transform(master_dataframe['mid'])
        master_dataframe['Low'] = lab_enc.fit_transform(master_dataframe['mid'])
        master_dataframe['Close'] = lab_enc.fit_transform(master_dataframe['mid'])
        train, test = train_test_split(master_dataframe, test_size=0.2)

        return train, test


def train_test():
    train, test = get_historical_data()
    model = load_model()
    fit_data(train, test, model)

def fit_data(train, test, SVM):
    print("Fitting Data")
    target_column_train = ['Close']
    predictors_train = list(set(list(train.columns)) - set(target_column_train))

    X_train = train[predictors_train].values
    y_train = train[target_column_train].values

    target_column_test = ['Close']
    predictors_test = list(set(list(test.columns)) - set(target_column_test))

    X_test = test[predictors_test].values
    y_test = test[target_column_test].values


    classifier = SVM
    classifier.verbose = 3
    classifier.fit(X_train, y_train.ravel())

    y_predict = classifier.predict(X_test)

    evaluation(y_test, y_predict)

    save_model(classifier)

def evaluation(y_test, y_predict):
    print("Classification Report below")
    print(classification_report(y_test, y_predict))


def stream_live_data(instruments = "EUR_USD"):
    #instruments = "DE30_EUR,EUR_USD,EUR_JPY"
    s = PricingStream(accountID=accountID, params={"instruments":instruments})
    try:
        index = 0
        for R in api.request(s):
            #print(json.dumps(R, indent=2))
            #print(R['instrument'])
            type = R['type']
            time = R['time']
            #print(R)
            if type == 'PRICE':

                instrument = R['instrument']
                close_bid = R['closeoutBid']
                print("{}: {} - {}".format(time, instrument, close_bid))
            else:
                print("Pulse: {}".format(time))
            #data = json.dumps(R, index=2)
            #print("{}".format(data['instrument']))
            index += 1
            if index > 10:
                print("Exiting loop, index {}".format(index))

    except V20Error as e:
        print("Error: {}".format(e))