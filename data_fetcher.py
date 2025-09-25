import yfinance as yf
from indicators import add_technical_indicators
import streamlit as st

@st.cache_data
def fetch_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = add_technical_indicators(df)
    return df

@st.cache_data
def fetch_live_data(ticker, period="1y"):
    df = yf.download(ticker, period=period)
    df = add_technical_indicators(df)
    return df
