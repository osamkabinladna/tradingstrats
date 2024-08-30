import pandas as pd
import re

def does_not_end_with_number(column_name):
    return not re.search(r'\d$', column_name)

def split_dataset(df):
    """
    Splits dataset per ticker
    :param df: dataframe to be split
    :param features: number of features for each ticker
    :return: list of dataframe split per ticker
    """
    tickerlist = []
    count = sum(does_not_end_with_number(col) for col in df.columns)
    features = count
    num_features_per_ticker = features  # Assuming each ticker has 11 features

    for i in range(int(len(df.columns) / features)):  # Assuming there are 27 tickers
        start = i * num_features_per_ticker
        end = start + num_features_per_ticker

        ticker = df.iloc[:, start:end]

        # If this is the first iteration, save the column names
        if i == 0:
            colnames = ticker.columns.tolist()

        # Set the columns to be the same for each ticker
        ticker.columns = colnames

        tickerlist.append(ticker)

    return tickerlist

def compute_covariates(cov):
    # Volume = Turnover / Close Price
    cov['VOLUME'] = cov['TURNOVER'] / cov['PX_LAST']
    # Calculate percent change * 100
    cov['PCT_CHANGE_20'] = ((cov['PX_LAST'].shift(-20) - cov['PX_LAST']) / cov['PX_LAST']) * 100
    # Ratio 10/30 = mean volume ratio for the last 10 days / mean volume ratio for the last 30 days
    cov['VOL_RATIO_10_20'] = cov['VOLUME'].rolling(window=10).mean() / cov['VOLUME'].rolling(window=20).mean()
    cov['VOL_RATIO_20_40'] = cov['VOLUME'].rolling(window=20).mean() / cov['VOLUME'].rolling(window=40).mean()
    cov['VOL_RATIO_40_80'] = cov['VOLUME'].rolling(window=40).mean() / cov['VOLUME'].rolling(window=80).mean()
    cov['VOL_RATIO_80_120'] = cov['VOLUME'].rolling(window=80).mean() / cov['VOLUME'].rolling(window=120).mean()

    # PE Band
    cov['PE_Ratio'] = 1 / cov['EARN_YLD']
    win = 60  # Set the rolling window period
    cov['PE_Band_25'] = cov['PE_Ratio'].rolling(win).quantile(0.25)
    cov['PE_Band_50'] = cov['PE_Ratio'].rolling(win).quantile(0.50)
    cov['PE_Band_75'] = cov['PE_Ratio'].rolling(win).quantile(0.75)

    # Calculate the 12-day EMA of PX_LAST
    ema_12 = cov['PX_LAST'].ewm(span=12, adjust=False).mean()

    # Calculate the 26-day EMA of PX_LAST
    ema_26 = cov['PX_LAST'].ewm(span=26, adjust=False).mean()

    # Calculate MACD
    cov['MACD'] = ema_12 - ema_26

    # Calculate the Signal line (9-day EMA of MACD)
    cov['MACD_Signal'] = cov['MACD'].ewm(span=9, adjust=False).mean()

    # Optionally, you can also calculate the MACD Histogram (the difference between MACD and Signal line)
    cov['MACD_Histogram'] = cov['MACD'] - cov['MACD_Signal']

    # Example: Momentum Indicator for various lags
    lags = [10, 20, 30, 60, 120]
    for lag in lags:
        cov[f'MOMENTUM_{lag}'] = cov['PX_LAST'] / cov['PX_LAST'].shift(lag)
        cov[f'TURNOVER_{lag}'] = cov['TURNOVER'].rolling(window=lag).mean()
        cov[f'PX_MOMENTUM_{lag}'] = cov['PX_LAST'] / cov['PX_LAST'].shift(lag)
        cov[f'PX_REVERSAL_{lag}'] = cov['PX_LAST'].shift(lag) / cov['PX_LAST']
        cov[f'VOLATILITY_{lag}'] = cov['PX_LAST'].rolling(window=lag).std()
        cov[f'VOLUME_STD_{lag}'] = cov['VOLUME'].rolling(window=lag).std()

    return cov