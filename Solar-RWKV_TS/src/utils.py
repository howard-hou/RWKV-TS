import json, time, random, os
import numpy as np
import dataclasses
from torch.nn import functional as F
from typing import List, Dict
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt


def plot_prediction_and_target(outputs, output_dir):
    for i in range(len(outputs)):
        plt.figure()
        pred = outputs[i]["predicts"].cpu().float().numpy()
        target = outputs[i]["targets"].cpu().float().numpy()
        x = np.arange(len(pred))
        plt.plot(x, pred, label="prediction", zorder=2)
        plt.plot(x, target, label="target", zorder=1)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"day_{i}.png"))
        plt.close()
    # plot week prediction
    week_outputs = [outputs[i:i+7] for i in range(0, len(outputs), 7)]
    week_pred = [np.concatenate([x["predicts"].cpu().float().numpy() for x in week], axis=0) for week in week_outputs]
    week_target = [np.concatenate([x["targets"].cpu().float().numpy() for x in week], axis=0) for week in week_outputs]
    for i in range(len(week_pred)):
        plt.figure()
        x = np.arange(len(week_pred[i]))
        plt.plot(x, week_pred[i], label="prediction", zorder=2)
        plt.plot(x, week_target[i], label="target", zorder=1)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"week_{i}.png"))
        plt.close()
    # plot month prediction
    month_outputs = [outputs[i:i+30] for i in range(0, len(outputs), 30)]
    month_pred = [np.concatenate([x["predicts"].cpu().float().numpy() for x in month], axis=0) for month in month_outputs]
    month_target = [np.concatenate([x["targets"].cpu().float().numpy() for x in month], axis=0) for month in month_outputs]
    for i in range(len(month_pred)):
        plt.figure()
        x = np.arange(len(month_pred[i]))
        plt.plot(x, month_pred[i], label="prediction", zorder=2)
        plt.plot(x, month_target[i], label="target", zorder=1)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"month_{i}.png"))
        plt.close()

