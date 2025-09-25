import streamlit as st
import pandas as pd
from data_fetcher import fetch_stock_data
from model_utils import scale_data, create_sequences, build_lstm_model
from predictions import generate_future_predictions, inverse_transform
from plots import plot_actual_vs_predicted

st.set_page_config(page_title="Ultimate Stock Predictor", layout="wide")
st.title("🚀 Ultimate Stock Price Predictor")

# Sidebar inputs
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
seq_len = st.sidebar.number_input("Sequence Length", 30, 120, 60)
future_days = st.sidebar.number_input("Days to Predict", 1, 30, 5)

# Load data
data = fetch_stock_data(ticker, start_date, end_date)
st.subheader(f"{ticker} Data")
st.dataframe(data.tail())

# Features and scaling
features = ['Open','High','Low','Close','Volume','MA10','MA50','EMA20','RSI','MACD','BB_H','BB_L']
scaled_data, scaler = scale_data(data, features)
X, y = create_sequences(scaled_data, seq_len)

# Build & train model
model = build_lstm_model((X.shape[1], X.shape[2]))
with st.spinner("Training model... ⏳"):
    model.fit(X, y, epochs=15, batch_size=32, verbose=0)

# Predict future
last_seq = scaled_data[-seq_len:]
predictions_scaled = generate_future_predictions(model, last_seq, future_days, len(features))
predictions = inverse_transform(scaler, predictions_scaled, features.index('Close'))

future_dates = pd.date_range(start=data.index[-1]+pd.Timedelta(days=1), periods=future_days)
st.subheader(f"Next {future_days} Days Predicted Close Prices")
st.line_chart(pd.DataFrame(predictions, index=future_dates, columns=['Predicted Close']))

# Plot actual vs predicted for last 60 days
predicted_past_scaled = model.predict(X)
predicted_past = inverse_transform(scaler, predicted_past_scaled, features.index('Close'))
fig = plot_actual_vs_predicted(data['Close'].values[-len(predicted_past):], predicted_past.flatten(), data.index[-len(predicted_past):])
st.plotly_chart(fig, use_container_width=True)

# Download predictions
st.download_button(
    label="Download Predictions CSV",
    data=pd.DataFrame(predictions, index=future_dates, columns=['Predicted Close']).to_csv().encode('utf-8'),
    file_name=f"{ticker}_predictions.csv",
    mime="text/csv"
)
