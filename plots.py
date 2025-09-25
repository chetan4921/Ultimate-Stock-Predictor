import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_actual_vs_predicted(actual, predicted, dates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Actual Close'))
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode='lines', name='Predicted Close'))
    return fig

def plot_future_predictions(predictions, future_dates):
    plt.figure(figsize=(10,4))
    plt.plot(future_dates, predictions, label='Predicted Close', color='green')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Future Predictions')
    return plt
