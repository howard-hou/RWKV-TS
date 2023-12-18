import matplotlib.pyplot as plt
import pandas as pd

def plot_multivariate_time_series(pred_df: pd.DataFrame, true_df: pd.DataFrame, title: str, proj_dir: str, plot_len: int = 96):
    """
    Plots a multivariate time series using matplotlib.

    Args:
        df (pd.DataFrame): The dataframe containing the time series data.
        title (str): The title of the plot.
    """
    n_features = pred_df.shape[1]
    # plot many subplot on one figure
    fig, axs = plt.subplots(n_features, 1, figsize=(10, 4*n_features))
    if n_features == 1:
        axs = [axs]
    fig.suptitle(title, fontsize=20)
    for i in range(n_features):
        axs[i].plot(pred_df.iloc[:, i][-plot_len:], label="Predicted")
        axs[i].plot(true_df.iloc[:, i][-plot_len:], label="True")
        axs[i].legend(loc="upper left")
        axs[i].grid()
    plt.savefig(f"{proj_dir}/{title}.png")


def plot_univariate_time_series(list_pred_df: pd.DataFrame, list_true_df: pd.DataFrame, title: str, proj_dir: str, plot_len: int = 96):
    """
    Plots a univariate time series using matplotlib.

    Args:
        df (pd.DataFrame): The dataframe containing the time series data.
        title (str): The title of the plot.
    """
    n_time_series = len(list_pred_df)
    # plot many subplot on one figure
    fig, axs = plt.subplots(n_time_series, 1, figsize=(10, 4*n_time_series))
    if n_time_series == 1:
        axs = [axs]
    fig.suptitle(title, fontsize=20)
    for i in range(n_time_series):
        axs[i].plot(list_pred_df[i].iloc[:, 0][-plot_len:], label="Predicted")
        axs[i].plot(list_true_df[i].iloc[:, 0][-plot_len:], label="True")
        axs[i].legend(loc="upper left")
        axs[i].grid()
    plt.savefig(f"{proj_dir}/{title}.png")