import streamlit as st
import io
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import re
from pathlib import Path
import os
import plotly.graph_objects as go

base_path = Path(__file__).resolve().parent.parent
models_path = base_path / 'random_forest' / 'models'

# Load the pre-trained model
@st.cache_resource
def load_model_and_data():
    model = joblib.load(models_path / 'yuge.joblib')
    valid_full = joblib.load(models_path / 'yuge_validfull.joblib')
    x_valid = joblib.load(models_path / 'yuge_xvalid.joblib')
    y_valid = joblib.load(models_path / 'yuge_yvalid.joblib')
    return model, valid_full, x_valid, y_valid

@st.cache_resource
def load_validation():
    x_valid = joblib.load('../random_forest/models/bigmodel_xvalid.joblib')
    y_valid = joblib.load('../random_forest/models/bigmodel_yvalid.joblib')
    return x_valid, y_valid

def does_not_end_with_number(column_name):
    return not re.search(r'\d$', column_name)

def split_dataset(df):
    tickerlist = []
    count = sum(does_not_end_with_number(col) for col in df.columns)
    features = count
    num_features_per_ticker = features

    for i in range(int(len(df.columns) / features)):
        start = i * num_features_per_ticker
        end = start + num_features_per_ticker

        ticker = df.iloc[:, start:end]
        if i == 0:
            colnames = ticker.columns.tolist()
        ticker.columns = colnames
        tickerlist.append(ticker)

    return tickerlist

def calculate_technical_indicators(df):
    df['VOLUME'] = df['TURNOVER'] / df['PX_LAST']
    df['PCT_CHANGE_20'] = ((df['PX_LAST'].shift(-20) - df['PX_LAST']) / df['PX_LAST']) * 100

    for short, long in [(10, 20), (20, 40), (40, 80), (80, 120)]:
        df[f'VOL_RATIO_{short}_{long}'] = df['VOLUME'].rolling(window=short).mean() / df['VOLUME'].rolling(window=long).mean()

    df['PE_Ratio'] = 1 / df['EARN_YLD']
    win = 60
    for percentile in [25, 50, 75]:
        df[f'PE_Band_{percentile}'] = df['PE_Ratio'].rolling(win).quantile(percentile / 100)

    ema_12 = df['PX_LAST'].ewm(span=12, adjust=False).mean()
    ema_26 = df['PX_LAST'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    lags = [10, 20, 30, 60, 120]
    for lag in lags:
        df[f'MOMENTUM_{lag}'] = df['PX_LAST'] / df['PX_LAST'].shift(lag)
        df[f'TURNOVER_{lag}'] = df['TURNOVER'].rolling(window=lag).mean()
        df[f'PX_MOMENTUM_{lag}'] = df['PX_LAST'] / df['PX_LAST'].shift(lag)
        df[f'PX_REVERSAL_{lag}'] = df['PX_LAST'].shift(lag) / df['PX_LAST']
        df[f'VOLATILITY_{lag}'] = df['PX_LAST'].rolling(window=lag).std()
        df[f'VOLUME_STD_{lag}'] = df['VOLUME'].rolling(window=lag).std()

    return df

def calculate_validation_stats(model, valid_full, x_valid, y_valid):
    predicted = model.predict(x_valid)
    predicted_probs = model.predict_proba(x_valid)

    valid_data = pd.DataFrame({
        'Ticker': valid_full['Ticker'],
        'Predicted': predicted,
        'Confidence': predicted_probs[:, 1],
        'Returns': valid_full['PCT_CHANGE_20'],
        'Buy Price': valid_full['PX_LAST'],
    })

    positive_preds = valid_data[valid_data["Predicted"] == 1]

    bins = range(50, 105, 5)
    labels = [f'{i}-{i+5}' for i in bins[:-1]]
    positive_preds['Probbin'] = pd.cut(positive_preds['Confidence'] * 100, bins=bins, labels=labels, right=False)

    stats_df = positive_preds.groupby('Probbin')['Returns'].agg(
        mean='mean',
        median='median',
        min='min',
        max='max',
        std='std',
        Winrate=lambda x: (x > 0).mean() * 100
    ).reset_index()

    # Initialize the plot
    fig = go.Figure()

    # Plot the PDF for each bin
    for label in labels:
        subset = positive_preds[positive_preds['Probbin'] == label]
        if not subset.empty:
            kde = subset['Returns'].plot.kde()
            x = kde.get_lines()[0].get_xdata()
            y = kde.get_lines()[0].get_ydata()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', fill='tozeroy', name=f'Prob Bin: {label}'))
            kde.remove()

    # Customize the layout
    fig.update_layout(
        title='PDF of Returns by Probability Bin',
        xaxis_title='Return',
        yaxis_title='Density',
        template='plotly_dark',
        legend_title_text='Probability Bins'
    )

    # Embed the plot in Streamlit
    st.plotly_chart(fig)

    return valid_data, stats_df

def plot_pdf_by_bin_plotly(positive_preds, bins, labels):
    positive_preds['Prob_Bin'] = pd.cut(positive_preds['Confidence'] * 100, bins=bins, labels=labels, right=False)
    fig = go.Figure()

    for label in labels:
        subset = positive_preds[positive_preds['Prob_Bin'] == label]
        if not subset.empty:
            kde = subset['Returns'].plot.kde()
            x = kde.get_lines()[0].get_xdata()
            y = kde.get_lines()[0].get_ydata()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', fill='tozeroy', name=f'Prob Bin: {label}'))
            kde.remove()

    fig.update_layout(
        title='PDF of Returns by Probability Bin',
        xaxis_title='Return',
        yaxis_title='Density',
        template='plotly_dark',
        legend_title_text='Probability Bins'
    )

    st.plotly_chart(fig)

def prepare_data(uploaded_file):
    file_content = uploaded_file.read()
    evaluation_data = pd.read_csv(io.BytesIO(file_content), skiprows=5, index_col='Dates')
    evaluation_data.index = pd.to_datetime(evaluation_data.index)

    evlist = split_dataset(evaluation_data)
    colnames = evlist[0].columns.tolist()
    for ev in evlist:
        ev.columns = colnames

    tickernames = pd.read_csv(io.BytesIO(file_content)).loc[2, :].dropna().tolist()
    for cov in evlist:
        calculate_technical_indicators(cov)

    evlist2b = [cov.tail(20) for cov in evlist]
    for i, df in enumerate(evlist2b):
        df.loc[:, 'Ticker'] = tickernames[i]

    pred_data = pd.concat(evlist2b, axis=0)
    return pred_data

def run_prediction(model, pred_data):
    pred_nopct = pred_data.drop('PCT_CHANGE_20', axis=1)
    pred_nona = pred_nopct.dropna(axis=0)

    predictions = model.predict(pred_nona.drop(['Ticker'], axis=1))
    prediction_probs = model.predict_proba(pred_nona.drop(['Ticker'], axis=1))

    preds = pd.DataFrame({
        'Predicted': predictions,
        'Confidence': prediction_probs[:, 1],
    })
    preds.index = pred_nona.index

    pred_nona['Predicted'] = preds['Predicted']
    pred_nona['Confidence'] = preds['Confidence']
    return pred_nona.loc[:, ['Ticker', 'Predicted', 'Confidence']]

def main():
    st.title("G Money App 🔮💸")

    model, valid_full, x_valid, y_valid = load_model_and_data()

    uploaded_file = st.file_uploader("Upload your CSV file for inference", type=["csv"])

    if uploaded_file is not None:
        pred_data = prepare_data(uploaded_file)
        predictions = run_prediction(model, pred_data)

        st.title("Predictions")
        positive_preds = predictions[predictions['Predicted'] == 1]
        st.write(positive_preds.drop_duplicates(subset=['Confidence', 'Ticker']))


    st.title("Validation Statistics")
    valid_data, stats_df = calculate_validation_stats(model, valid_full, x_valid, y_valid)
    st.write("Statistics by Confidence Bin:")
    st.write(stats_df)

    y_pred = model.predict(x_valid)
    st.write(f"Accuracy: {accuracy_score(y_valid, y_pred)}")

    report_dict = classification_report(y_valid, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.write("Classification Report")
    st.dataframe(report_df)

if __name__ == "__main__":
    main()