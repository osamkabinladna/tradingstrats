import pandas as pd
import re
import os

class ExcelDataloader:
    def __init__(self, covariates_path, labels_path, train_window=252, test_window=60):
        self.covariates_path = covariates_path
        self.labels_path = labels_path
        self.train_window = train_window
        self.test_window = test_window
        self.covlist_reindex = []
        self.labels = None
        self.mode = None
        self.num_tickers = None

    def load_data(self):
        # Load covariates and labels
        covariates = pd.read_csv(self.covariates_path, index_col="Dates", skiprows=5)
        covariates.index = pd.to_datetime(covariates.index)

        labels = pd.read_csv(self.labels_path, skiprows=5, index_col="Dates")
        labels.index = pd.to_datetime(labels.index)
        labels = labels.rename(columns={"PX_LAST": "JKSE_PRICE"})
        labels['PCT_CHANGE_20_JKSE'] = ((labels['JKSE_PRICE'].shift(-20) - labels['JKSE_PRICE']) / labels['JKSE_PRICE']) * 100

        self.labels = labels

        # Determine the number of tickers by inspecting the column names
        tickernames = [col[:4] for col in covariates.columns if not col.startswith("Unnamed")]
        self.num_tickers = len(set(tickernames))

        # Split the dataset
        self.covlist_reindex = self.split_dataset(covariates)

        # Calculate additional features
        self.calculate_covariates()
        self.calculate_delta()

    def non_num_suffix(self, colname):
        return not re.search(r'\d$', colname)

    def split_dataset(self, df):
        """
        Splits dataset per ticker
        :param df: dataframe to be split
        :return: list of dataframe split per ticker
        """
        num_features_per_ticker = df.shape[1] // self.num_tickers
        tickerlist = []

        for i in range(self.num_tickers):
            start = i * num_features_per_ticker
            end = start + num_features_per_ticker
            ticker = df.iloc[:, start:end]
            tickerlist.append(ticker)

        return tickerlist

    def calculate_covariates(self):
        tickernames = [col[:4] for col in pd.read_csv(self.covariates_path, skiprows=3).columns if not col.startswith("Unnamed")]

        for i, cov in enumerate(self.covlist_reindex):
            cov['Ticker'] = tickernames[i]

            if 'TURNOVER' in cov.columns and 'PX_LAST' in cov.columns:
                cov['VOLUME'] = cov['TURNOVER'] / cov['PX_LAST']
                cov['PCT_CHANGE_20'] = ((cov['PX_LAST'].shift(-20) - cov['PX_LAST']) / cov['PX_LAST']) * 100
                cov['VOL_RATIO_10_20'] = cov['VOLUME'].rolling(window=10).mean() / cov['VOLUME'].rolling(window=20).mean()
                cov['VOL_RATIO_20_40'] = cov['VOLUME'].rolling(window=20).mean() / cov['VOLUME'].rolling(window=40).mean()
                cov['VOL_RATIO_40_80'] = cov['VOLUME'].rolling(window=40).mean() / cov['VOLUME'].rolling(window=80).mean()
                cov['VOL_RATIO_80_120'] = cov['VOLUME'].rolling(window=80).mean() / cov['VOLUME'].rolling(window=120).mean()
            else:
                print(f"Warning: 'TURNOVER' or 'PX_LAST' column not found in ticker {tickernames[i]}.")

            cov['PE_Ratio'] = 1 / cov['EARN_YLD']
            win = 60
            cov['PE_Band_25'] = cov['PE_Ratio'].rolling(win).quantile(0.25)
            cov['PE_Band_50'] = cov['PE_Ratio'].rolling(win).quantile(0.50)
            cov['PE_Band_75'] = cov['PE_Ratio'].rolling(win).quantile(0.75)

            ema_12 = cov['PX_LAST'].ewm(span=12, adjust=False).mean()
            ema_26 = cov['PX_LAST'].ewm(span=26, adjust=False).mean()
            cov['MACD'] = ema_12 - ema_26
            cov['MACD_Signal'] = cov['MACD'].ewm(span=9, adjust=False).mean()
            cov['MACD_Histogram'] = cov['MACD'] - cov['MACD_Signal']

            lags = [10, 20, 30, 60, 120]
            for lag in lags:
                cov[f'MOMENTUM_{lag}'] = cov['PX_LAST'] / cov['PX_LAST'].shift(lag)
                cov[f'TURNOVER_{lag}'] = cov['TURNOVER'].rolling(window=lag).mean() if 'TURNOVER' in cov.columns else None
                cov[f'PX_MOMENTUM_{lag}'] = cov['PX_LAST'] / cov['PX_LAST'].shift(lag)
                cov[f'PX_REVERSAL_{lag}'] = cov['PX_LAST'].shift(lag) / cov['PX_LAST']
                cov[f'VOLATILITY_{lag}'] = cov['PX_LAST'].rolling(window=lag).std()
                cov[f'VOLUME_STD_{lag}'] = cov['VOLUME'].rolling(window=lag).std() if 'VOLUME' in cov.columns else None

    def calculate_delta(self):
        pooled_df = pd.DataFrame()
        for i in range(len(self.covlist_reindex)):
            cov = self.covlist_reindex[i]
            cov = cov[~cov.index.duplicated(keep='first')]
            cov_copy = cov.copy()

            aligned_df = self.labels.join(cov_copy[['PCT_CHANGE_20']], how='inner')
            cov_copy.loc[aligned_df.index, 'DELTA_20_CHANGE'] = aligned_df['PCT_CHANGE_20'] - aligned_df['PCT_CHANGE_20_JKSE']

            self.covlist_reindex[i] = cov_copy
            pooled_df = pd.concat([pooled_df, cov_copy['DELTA_20_CHANGE']])

        pooled_df['DELTA_20_QUINTILES'] = pd.qcut(pooled_df[0], q=5, labels=range(1, 6))

        for i, cov in enumerate(self.covlist_reindex):
            cov['DELTA_20_QUINTILES'] = pooled_df.loc[cov.index, 'DELTA_20_QUINTILES']
            cov['TOP_5'] = cov['DELTA_20_QUINTILES'].apply(lambda x: 1 if x == 5 else 0)
            cov['TOP_5'] = cov['TOP_5'].fillna(0).astype(int)
            cov.dropna(inplace=True)
            cov.reset_index(drop=False, inplace=True)
            self.covlist_reindex[i] = cov

    def rolling_window_save_oob(self):
        tdf = []
        vdf = []

        for cov in self.covlist_reindex:
            ticker = cov['Ticker'].iloc[0]
            num_windows = (len(cov) - self.train_window - self.test_window) // self.test_window + 1

            for i in range(num_windows):
                start_train = i * (self.train_window + self.test_window)
                end_train = start_train + self.train_window
                start_test = end_train
                end_test = start_test + self.test_window

                train_df = cov.iloc[start_train:end_train].copy()
                test_df = cov.iloc[start_test:end_test].copy()

                if train_df.empty or test_df.empty:
                    continue

                train_df['Ticker'] = ticker
                test_df['Ticker'] = ticker
                test_df['Window'] = i

                tdf.append(train_df)
                vdf.append(test_df)

        return tdf, vdf

    def get_datasets(self, mode='quintiles'):
        self.load_data()
        self.mode = mode.lower()
        tdf, vdf = self.rolling_window_save_oob()

        if self.mode == 'quintiles':
            train_data = pd.concat(tdf, ignore_index=True).drop(['TOP_5'], axis=1)
            valid_data = pd.concat(vdf, ignore_index=True).drop(['TOP_5'], axis=1)
        elif self.mode == 'top_5':
            train_data = pd.concat(tdf, ignore_index=True).drop(['DELTA_20_QUINTILES'], axis=1)
            valid_data = pd.concat(vdf, ignore_index=True).drop(['DELTA_20_QUINTILES'], axis=1)
        else:
            raise ValueError("Mode should be either 'quintiles' or 'top_5'.")

        return train_data, valid_data, vdf

# Usage
data_loader = ExcelDataloader(
    covariates_path="./data/bigdata.csv",
    labels_path="./data/jkse.csv"
)

train_data, valid_data, crossval_data = data_loader.get_datasets(mode='top_5')