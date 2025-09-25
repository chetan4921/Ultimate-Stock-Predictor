import ta
import pandas as pd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds common technical indicators to the stock DataFrame.
    Returns a DataFrame with added columns: MA10, MA50, EMA20, RSI, MACD, BB_H, BB_L
    """

    # Ensure 'Close' is 1D
    close = df['Close'].squeeze()

    # Moving Averages
    df['MA10'] = close.rolling(window=10).mean()
    df['MA50'] = close.rolling(window=50).mean()
    df['EMA20'] = close.ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # MACD
    macd_indicator = ta.trend.MACD(close)
    df['MACD'] = macd_indicator.macd()

    # Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(close)
    df['BB_H'] = bb_indicator.bollinger_hband()
    df['BB_L'] = bb_indicator.bollinger_lband()

    # Drop rows with NaN values created by rolling windows
    df = df.dropna()

    return df
