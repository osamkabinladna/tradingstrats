import streamlit as st
import pandas as pd
import ydf
from utils import compute_covariates
import os
import asyncio
import concurrent.futures

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
async def load_model_async():
    model_str = "../random_forest/models/100tree_100d"
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        model = await loop.run_in_executor(pool, ydf.load_model, model_str)
    return model

@st.cache_data
def load_and_process_csv(uploaded_file):
    df = pd.read_csv(uploaded_file, usecols=lambda column: True, nrows=10, skiprows=lambda i: i > 0 and i < (sum(1 for line in uploaded_file) - 11))
    return df

async def main():
    st.title("G Money")

    # Start loading the model asynchronously
    model_future = asyncio.create_task(load_model_async())

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load and process the CSV file
        df = load_and_process_csv(uploaded_file)

        # Compute covariates
        covpred = compute_covariates(df)

        # Wait for the model to load, with a loading indicator
        with st.spinner('Loading model...'):
            try:
                model = await asyncio.wait_for(model_future, timeout=30)  # Increased timeout to 30 seconds
            except asyncio.TimeoutError:
                st.error("Model loading timed out. Please try again.")
                return

        if model is not None:
            # Make predictions
            crossval_preds = model.predict(covpred)

            # Set threshold for classification
            threshold = 0.5
            predicted_classes = (crossval_preds >= threshold).astype(int)

            # Create a DataFrame with predictions
            preds = pd.DataFrame({
                'Predicted': predicted_classes,
                'Probs': crossval_preds,
                "Dates": covpred["Dates"]
            })

            # Align predictions with covariates
            preds.index = covpred.index
            covpred['Predicted'] = preds['Predicted']
            covpred['Confidence'] = preds['Probs']

            # Filter positive predictions
            crossvalpred_data = covpred.loc[:, ['Ticker', 'Predicted', "Confidence", "Dates"]]
            positive_preds = crossvalpred_data[crossvalpred_data["Predicted"] == 1]

            # Display the positive predictions
            st.write(positive_preds)
    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    asyncio.run(main())