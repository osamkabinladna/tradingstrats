import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from hmmlearn import hmm
from sklearn.metrics import akaike_information_criterion as aic

class FinancialDataProcessor:
    def __init__(self, filepath, delimiter=";", date_threshold='2010-10-10', training_ratio=0.8):
        self.filepath = filepath
        self.delimiter = delimiter
        self.date_threshold = pd.to_datetime(date_threshold)
        self.training_ratio = training_ratio
        self.df_adj_close = None
        self.df_volume = None
        self.train_df = None
        self.valid_df = None
        self.cov_matrix = None
        self.pca = None
        self.cumulative_variance_ratio = None
        self.reduced_data = None
        self.best_hmm = None

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.filepath, delimiter=self.delimiter, index_col=0)
        filtered_colnames = [col[:4] for col in df.columns.tolist() if "Unnamed" not in col]

        df = pd.read_csv(self.filepath, delimiter=self.delimiter, header=1, index_col=0)
        df = df[pd.to_datetime(df.index, errors="coerce").notna()]

        adj_close_cols = [col for col in df.columns if 'PX_LAST' in col]
        volume_cols = [col for col in df.columns if 'VOLUME' in col]

        self.df_adj_close = df[adj_close_cols].apply(pd.to_numeric, errors='coerce')
        self.df_volume = df[volume_cols].apply(pd.to_numeric, errors='coerce')

        self.df_adj_close.index = pd.to_datetime(self.df_adj_close.index)
        self.df_adj_close.columns = filtered_colnames
        self.df_volume.index = pd.to_datetime(self.df_volume.index)
        self.df_volume.columns = filtered_colnames

        self.df_adj_close = self.df_adj_close.resample('W').last().sort_index(ascending=False)
        self.df_adj_close = self.df_adj_close[self.df_adj_close.index >= self.date_threshold].dropna(axis=1)

    def calculate_returns(self):
        df_returns = self.df_adj_close.pct_change().iloc[1:]
        returns_norm = (df_returns - df_returns.mean()) / df_returns.std()

        split_point = int(np.round(returns_norm.shape[0] * self.training_ratio))
        self.train_df = returns_norm.iloc[:split_point]
        self.valid_df = returns_norm.iloc[split_point:]

        self.cov_matrix = self.train_df.cov()

    def plot_cov_matrix(self):
        mask = np.triu(np.ones_like(self.cov_matrix, dtype=bool))

        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(self.cov_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()

    def pca_analysis(self, p=0.30):
        eigenvalues, eigenvectors = np.linalg.eig(self.cov_matrix)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_variance = np.sum(eigenvalues)
        cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance

        k = np.argmax(cumulative_variance_ratio >= (1 - p)) + 1

        self.pca = PCA(n_components=k)
        self.reduced_data = self.pca.fit_transform(self.train_df)
        self.cumulative_variance_ratio = cumulative_variance_ratio

        return self.pca, self.cumulative_variance_ratio, self.reduced_data

    def train_and_select_hmm(self, factor_returns, max_states=8):
        best_model = None
        best_aic = np.inf

        for n_states in range(2, max_states + 1):
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)

            try:
                model.fit(factor_returns)
                model_aic = aic(model.score(factor_returns), n_params=model.n_parameters())

                if model_aic < best_aic:
                    best_aic = model_aic
                    best_model = model
            except:
                continue

        self.best_hmm = best_model
        return best_model

    def forecast_returns(self, pca, hmm_model, normalized_returns):
        last_factors = pca.transform(normalized_returns.iloc[-1:])

        next_state = hmm_model.predict(last_factors)[-1]

        state_mean = hmm_model.means_[next_state]

        forecasted_normalized = pca.inverse_transform(state_mean.reshape(1, -1))

        forecasted_returns = forecasted_normalized * normalized_returns.std() + normalized_returns.mean()

        return forecasted_returns.flatten()

    def implement_trading_strategies(self, returns, normalized_returns, window_size=520):
        strategy1_returns = []
        strategy2_returns = []

        for i in range(window_size, len(returns)):
            train_normalized = normalized_returns.iloc[i-window_size:i]

            self.pca.fit(train_normalized)
            factor_returns = self.pca.transform(train_normalized)
            self.best_hmm.fit(factor_returns)

            forecasted_returns = self.forecast_returns(self.pca, self.best_hmm, train_normalized)
            forecasted_normalized = (forecasted_returns - returns.iloc[i-1].mean()) / returns.iloc[i-1].std()

            strategy1_position = np.sign(forecasted_returns)
            strategy1_return = np.sum(strategy1_position * returns.iloc[i]) / len(returns.columns)
            strategy1_returns.append(strategy1_return)

            strategy2_position = np.sign(forecasted_normalized)
            strategy2_return = np.sum(strategy2_position * returns.iloc[i]) / len(returns.columns)
            strategy2_returns.append(strategy2_return)

        return pd.Series(strategy1_returns, index=returns.index[window_size:]), pd.Series(strategy2_returns, index=returns.index[window_size:])

    def evaluate_performance(self, returns, strategy_returns):
        winning_prob = np.mean(strategy_returns > 0)

        sharpe_ratio = np.sqrt(52) * strategy_returns.mean() / strategy_returns.std()

        cumulative_returns = (1 + strategy_returns).cumprod()

        return winning_prob, sharpe_ratio, cumulative_returns

    def plot_cumulative_returns(self, strategy1_metrics, strategy2_metrics, bnh_metrics):
        plt.figure(figsize=(12, 6))
        plt.plot(strategy1_metrics[2], label='Strategy 1')
        plt.plot(strategy2_metrics[2], label='Strategy 2')
        plt.plot(bnh_metrics[2], label='Buy-and-Hold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.title('Cumulative Returns Comparison')
        plt.legend()
        plt.show()
