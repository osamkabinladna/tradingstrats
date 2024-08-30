import pandas as pd
import numpy as np
import joblib

def load_model_and_data(model_path, valid_full_path, x_valid_path, y_valid_path):
    model = joblib.load(model_path)
    valid_full = joblib.load(valid_full_path)
    x_valid = joblib.load(x_valid_path)
    y_valid = joblib.load(y_valid_path)
    return model, valid_full, x_valid, y_valid

def create_validation_dataframe(model, valid_full, x_valid):
    predicted = model.predict(x_valid)
    predicted_probs = model.predict_proba(x_valid)

    valid_data = pd.DataFrame({
        'Ticker': valid_full['Ticker'],
        'Predicted': predicted,
        'Confidence': predicted_probs[:,1],
        'Returns': valid_full['PCT_CHANGE_20']
    })
    return valid_data

def analyze_positive_predictions(valid_data):
    positive_preds = valid_data[valid_data["Predicted"] == 1]

    bins = range(50, 105, 5)
    labels = [f'{i}-{i+5}' for i in bins[:-1]]

    positive_preds['Prob Bin'] = pd.cut(positive_preds['Confidence'] * 100, bins=bins, labels=labels, right=False)

    stats_df = positive_preds.groupby('Prob Bin')['Returns'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()

    return stats_df

def plot_returns_distribution(positive_preds):
    import plotly.graph_objects as go

    bins = range(50, 105, 5)
    labels = [f'{i}-{i+5}' for i in bins[:-1]]
    positive_preds['Prob Bin'] = pd.cut(positive_preds['Confidence'] * 100, bins=bins, labels=labels, right=False)

    fig = go.Figure()

    for label in labels:
        subset = positive_preds[positive_preds['Prob Bin'] == label]
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
        template='plotly_white',
        legend_title_text='Probability Bins'
    )

    return fig