import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_pred(x, y_pred, y_true, exp_name, output_path):
    """Plot a single prediction."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()

    plt.figure(figsize=(8, 6), dpi=300)
    x_pred = np.arange(len(x), len(x)+len(y_pred))
    # concat all x and y_true
    x_true = np.arange(len(x), len(x)+len(y_true))
    plt.plot(x, label='Input', color='blue')
    plt.plot(x_true, y_true, label='Truth', color='black')
    plt.plot(x_pred, y_pred, label=exp_name, color='purple')
    # save
    plt.legend()
    plt.savefig(output_path)

def visualize_experiment(exp, output_dir, col=0, k=1, num_plots=6):
    """Visualize a single experiment."""
    # random select num_plots index from test dataset
    num_total_samples = len(exp.test_dataset)
    indices = np.random.choice(num_total_samples, num_plots, replace=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_name = output_dir.name
    for i in tqdm(indices, desc='visualizing'):
        output_path = output_dir / f'{i}.png'
        one_pred = exp.run_one_univariate_predict(i=i, col=col, k=k)
        if one_pred is None:
            continue
        x, y_pred, y_true = one_pred
        plot_pred(x, y_pred, y_true, exp_name=exp_name, output_path=output_path)
