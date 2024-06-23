import numpy as np
import plotly.express as px

def load_house_data():
    """
    Load house data from a text file.

    Returns:
    X : numpy.ndarray
        Array of feature values (size, bedrooms, bathrooms, age).
    y : numpy.ndarray
        Array of target values (price).
    """
    data = np.loadtxt("houses.txt", delimiter=',', skiprows=1)
    X = data[:, :4]
    y = data[:, 4]
    return X, y

def scatter_plot_prices(df):
    """
    Create a scatter plot of house prices.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing 'size' and 'price' columns.

    Returns:
    fig : plotly.graph_objs.Figure
        Plotly figure object of the scatter plot.
    """
    fig = px.scatter(df, x='size', y='price')
    fig.update_layout(
        title={
            'text': 'House prices',
            'x': 0.5
        },
        xaxis_title='Size (ft²)',
        yaxis_title='Price (k$)',
    )

    fig.update_traces(
        marker=dict(
            symbol='x',
            color='red',
            size=15
        ))

    fig.write_image("size_price.png")

    return fig

def scatter_plot_cost(J_history):
    """
    Create a scatter plot of the cost function evolution.

    Parameters:
    J_history : list or numpy.ndarray
        List or array containing the history of cost function values.

    Returns:
    fig : plotly.graph_objs.Figure
        Plotly figure object of the scatter plot.
    """
    fig = px.scatter(x=range(len(J_history)), y=J_history)
    fig.update_layout(
        title={
            'text': "Cost function evolution",
            'x': 0.5
        },
        xaxis_title='Iteration number',
        yaxis_title='Cost'
    )

    return fig

def plot_line_fit(df, w, b):
    """
    Plot the fitted line on top of a scatter plot of house prices.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing 'size' and 'price' columns.
    w : float
        Slope of the fitted line.
    b : float
        Intercept of the fitted line.

    Returns:
    fig : plotly.graph_objs.Figure
        Plotly figure object with the scatter plot and fitted line.
    """
    fig = scatter_plot_prices(df)

    x_line = np.array([np.min(df['size']), np.max(df['size'])])
    y_line = w * x_line + b

    fig.add_scatter(x=x_line, y=y_line, mode='lines', name=f'f = {w:.3f}x + {b:.6f}')

    return fig

def plot_fit_evolution(df, p_history):
    """
    Plot the evolution of fitted lines on top of a scatter plot of house prices.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing 'size' and 'price' columns.
    p_history : list of tuples
        List of tuples, where each tuple contains (w, b) representing slope and intercept of a fitted line.

    Returns:
    fig : plotly.graph_objs.Figure
        Plotly figure object with the scatter plot and all fitted lines from p_history.
    """
    fig = scatter_plot_prices(df)

    x_line = np.array([np.min(df['size']), np.max(df['size'])])
    for w, b in p_history:
        y_line = w * x_line + b
        fig.add_scatter(x=x_line, y=y_line, mode='lines', showlegend=False)

    return fig
