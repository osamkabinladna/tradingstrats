import pandas as pd


def split_dataset(df, features = 12):
    """
    Splits dataset per ticker
    :param df: dataframe to be split
    :param features: number of features for each ticker
    :return: list of dataframe split per ticker
    """
    tickerlist = []
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

def calculate_delta(labels: pd.DataFrame, covariates_list: list):
    """
    Calculates the delta from label, function is adapted to handle missing labels
    :param labels: label dataframe
    :param covariates_list: list of dataframes containing covariates
    :return: covariates list
    """

    # Remove duplicate indices if any
    labels = labels[~labels.index.duplicated(keep='first')]

    for i in range(len(covariates_list)):
        cov = covariates_list[i]
        cov = cov[~cov.index.duplicated(keep='first')]

        # copy a copy of the cov dataframe to avoid settingwithcopywarning
        cov_copy = cov.copy()

        aligned_df = labels.join(cov_copy[['PCT_CHANGE_20']], how='inner', lsuffix='labels')
        cov_copy.loc[aligned_df.index, 'DELTA_20_CHANGE'] = aligned_df['PCT_CHANGE_20'] - aligned_df['PCT_CHANGE_20_JKSE']

        covariates_list[i] = cov_copy

    return covariates_list, aligned_df

def calculate_delta_fix(labels: pd.DataFrame, covariates_list: list):
    """
    Calculates the delta from label, function is adapted to handle missing labels
    :param labels: label dataframe
    :param covariates_list: list of dataframes containing covariates
    :return: covariates list
    """

    # Remove duplicate indices if any

    for i in range(len(covariates_list)):
        cov = covariates_list[i]
        cov = cov[~cov.index.duplicated(keep='first')]

        # Copy the cov dataframe to avoid SettingWithCopyWarning
        cov_copy = cov.copy()

        # Ensure that the indices (dates) are aligned
        aligned_df = labels.join(cov_copy[['PCT_CHANGE_20']], how='inner')

        # Debugging step: check if aligned_df is empty
        if aligned_df.empty:
            print(f"aligned_df is empty for index {i}. Inspecting indices:")
            print("Labels index range:", labels.index.min(), labels.index.max())
            print("Covariates index range:", cov_copy.index.min(), cov_copy.index.max())

        else:
            cov_copy.loc[aligned_df.index, 'DELTA_20_CHANGE'] = aligned_df['PCT_CHANGE_20_JKSE'] - aligned_df['PCT_CHANGE_20']

        covariates_list[i] = cov_copy

    return covariates_list, aligned_df