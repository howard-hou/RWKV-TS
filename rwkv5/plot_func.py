import matplotlib.pyplot as plt
import pandas as pd

def plot_multivariate_time_series(pred_df: pd.DataFrame, true_df: pd.DataFrame, title: str, proj_dir: str):
    """
    Plots a multivariate time series using matplotlib.

    Args:
        df (pd.DataFrame): The dataframe containing the time series data.
        title (str): The title of the plot.
    """
    n_features = pred_df.shape[1]
    # plot many subplot on one figure
    fig, axs = plt.subplots(n_features, 1, figsize=(20, 30))
    if n_features == 1:
        axs = [axs]
    fig.suptitle(title, fontsize=20)
    for i in range(n_features):
        axs[i].plot(pred_df.iloc[:, i], label="Predicted")
        axs[i].plot(true_df.iloc[:, i], label="True")
        axs[i].legend(loc="upper left")
        axs[i].grid()
    plt.savefig(f"{proj_dir}/{title}.png")