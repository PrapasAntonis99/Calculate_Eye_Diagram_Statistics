import os
import time
import math
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from matplotlib.figure import Figure
from scipy.signal import argrelextrema
from sklearn.mixture import GaussianMixture


def check_directory_exists(directory, root='Results'):
    dir_path = os.getcwd()
    dir_path = dir_path.replace('\\', '/')

    if root == 'Results':
        data_dir = dir_path + '/' + root + '/' + directory
    else:
        data_dir = dir_path + '/' + root + '/'

    already_exists = os.path.exists(data_dir)

    if not already_exists:
        os.makedirs(data_dir)
        if root != 'Results':
            exit('Empty folder')


def check_file_exists(directory, file):
    dir_path = os.getcwd()
    dir_path = dir_path.replace('\\', '/')

    file_csv = os.path.isfile(dir_path + '/' + directory + '/' + file + '.csv')
    file_pkl = os.path.isfile(dir_path + '/' + directory + '/' + file + '.pkl')

    if file_csv:
        return pd.read_csv(dir_path + '/' + directory + '/' + file + '.csv')
    elif file_pkl:
        return pd.read_csv(dir_path + '/' + directory + '/' + file + '.pkl')
    else:
        exit('No data file (csv or pkl)')


def upsampler(input_signal, upsample_factor):
    upsampled_signal = np.repeat(input_signal, upsample_factor)
    return upsampled_signal


def downsampler(input_signal, downsample_factor):
    downsampled_signal = np.zeros(len(input_signal) // downsample_factor)
    j = 0
    for i in range(len(input_signal) - downsample_factor + 1):
        if i % downsample_factor == 0:
            downsampled_signal[j] = input_signal[i]
            j += 1
    return downsampled_signal


def cut_signal_samples(signal, cut_last_symbols, updated_sps):
    new_length = len(signal) - cut_last_symbols * updated_sps
    return signal[:new_length]


def plot_signal(signal):
    plt.title('Recorded Signal')
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.plot(signal)
    plt.show()


def bres_segment_count(x0, y0, x1, y1, grid):
    """Bresenham's algorithm.

    The value of grid[x,y] is incremented for each x,y
    in the line from (x0,y0) up to but not including (x1, y1).
    """

    if np.any(np.isnan([x0, y0, x1, y1])):
        return

    nrows, ncols = grid.shape

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 0
    if x0 < x1:
        sx = 1
    else:
        sx = -1

    sy = 0
    if y0 < y1:
        sy = 1
    else:
        sy = -1

    err = dx - dy

    while True:
        if x0 == x1 and y0 == y1:
            break

        if 0 <= x0 < nrows and 0 <= y0 < ncols:
            grid[int(x0), int(y0)] += 1

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def bres_curve_count(y, x, grid):
    for k in range(x.size - 1):
        x0 = x[k]
        y0 = y[k]
        x1 = x[k + 1]
        y1 = y[k + 1]
        bres_segment_count(x0, y0, x1, y1, grid)


def linear_scale(x, src_min, src_max, dst_min, dst_max):
    return dst_min + (x - src_min) * (dst_max - dst_min) / (src_max - src_min)


def fig_show(figure):
    for i in plt.get_fignums():
        if figure != plt.figure(i):
            plt.close(plt.figure(i))
    plt.show()


def draw_eye_diagram(input_signal, input_signal_period, calculate_window, file):
    check_directory_exists(file)

    symbol_slice = 2

    eye_center = input_signal_period
    upper_limit = int(eye_center + input_signal_period * (calculate_window / 2) / 100)
    lower_limit = int(eye_center - input_signal_period * (calculate_window / 2) / 100)

    number_of_symbols = len(input_signal) // input_signal_period
    repeats = number_of_symbols // symbol_slice

    x = np.linspace(0, input_signal_period * symbol_slice, input_signal_period * symbol_slice)
    x = repmat(a=x, m=repeats, n=1).flatten()

    bins = [input_signal_period * symbol_slice, input_signal_period]

    fig, ax = plt.subplots(dpi=200)
    plt.title('Eye Diagram')
    ax.hist2d(x, input_signal[:len(x)], bins, cmap=plt.cm.hot)
    ax.get_xaxis().set_visible(False)
    fig.savefig('Results' + '/' + file + '/' + 'eye diagram' + ' (' + str(input_signal_period) + ')' + '.png')

    fig, ax = plt.subplots()
    plt.title('Eye Diagram')
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    h, x_edges, y_edges, eye_print = ax.hist2d(x, input_signal[:len(x)], bins, cmap=plt.cm.hot)
    ax.axvline(eye_center, color='green', linestyle='--', linewidth=1)
    ax.axvline(upper_limit, color='green', linestyle='-', linewidth=1)
    ax.axvline(lower_limit, color='green', linestyle='-', linewidth=1)
    plt.colorbar(eye_print)
    fig_show(fig)


def calculate_q_factor(input_signal, input_signal_levels, input_signal_period, calculate_window, file):
    check_directory_exists(file)

    grid_W = 2000
    grid_H = 1200

    grid = np.zeros((grid_H, grid_W), dtype=np.int32)

    ys = []
    for i in range(0, len(input_signal) // (2 * input_signal_period)):
        ys.append(input_signal[int(i * input_signal_period * 2): int((i + 1) * input_signal_period * 2)])

    t_d = np.round(np.linspace(0, grid_W, input_signal_period * 2 + 1, dtype=np.int32))

    y_min = np.nanmin(ys)
    y_max = np.nanmax(ys)
    ys_d = []
    for y in ys:
        ys_d.append(np.round(linear_scale(y, y_min, y_max, 0, grid_H)))

    for y_d in ys_d:
        bres_curve_count(t_d, y_d, grid)

    eye_count = np.zeros(grid_H)

    upper_limit = int(grid_W // 2 + grid_W // 2 * (calculate_window / 2) / 100)
    lower_limit = int(grid_W // 2 - grid_W // 2 * (calculate_window / 2) / 100)
    eye_range = upper_limit - lower_limit

    for i in range(grid_H):
        for j in range(lower_limit, upper_limit):
            eye_count[i] += grid[i][j]
        eye_count[i] /= eye_range

    eye_center = []
    for i in range(len(eye_count)):
        limit = int(round(eye_count[i], 2) * 100)
        for j in range(limit):
            eye_center.append(i)

    eye_q_factor = np.array(eye_center).reshape(-1, 1)

    gmm = GaussianMixture(n_components=input_signal_levels)
    gmm.fit(eye_q_factor)

    levels = gmm.means_.flatten()
    covariances = np.sqrt(gmm.covariances_.flatten())

    sorted_indices = np.argsort(levels)
    levels = levels[sorted_indices]
    covariances = covariances[sorted_indices]

    print(f'Distribution Levels: {levels}')
    print(f'Distribution Covariances: {covariances}')

    q_factor = []
    for i in range(input_signal_levels - 1):
        value = round((levels[i + 1] - levels[i]) / (covariances[i + 1] + covariances[i]), 2)
        print(f'Q{i + 1}: {value}')
        q_factor.append(value)

    plt.figure(dpi=150)
    plt.title('Q Factor Distributions')
    graph = sns.histplot(eye_center, bins=max(eye_center), kde=True)
    for level in levels:
        plt.axvline(level, color='red', linestyle='--', linewidth=1)
    graph.set(yticklabels=[])
    plt.savefig('Results' + '/' + file + '/' + 'eye center distributions' + ' (' + str(input_signal_period) + ')' + '.png')

    q_factor = plt.figure(dpi=120)
    plt.title('Q Factor Distributions')
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    graph = sns.histplot(eye_center, bins=max(eye_center), kde=True)
    for level in levels:
        plt.axvline(level, color='red', linestyle='--', linewidth=1)
    graph.set(yticklabels=[])
    fig_show(q_factor)


def maximize_q_factor(input_signal, input_signal_levels, input_signal_period, roll_search_window, speed_up):
    symbol_slice = 2

    maximum_limit = int(input_signal_period + input_signal_period * (roll_search_window / 2) / 100)
    minimum_limit = int(input_signal_period - input_signal_period * (roll_search_window / 2) / 100)

    number_of_symbols = len(input_signal) // input_signal_period
    repeats = number_of_symbols // symbol_slice

    x = np.linspace(0, input_signal_period * symbol_slice, input_signal_period * symbol_slice)
    x = repmat(a=x, m=repeats, n=1).flatten()

    bins = [input_signal_period * symbol_slice, input_signal_period]

    fig, ax = plt.subplots()
    plt.title('Roll Period')
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    ax.hist2d(x, input_signal[:len(x)], bins, cmap=plt.cm.hot)
    ax.axvline(maximum_limit, color='blue', linestyle='-', linewidth=1)
    ax.axvline(minimum_limit, color='blue', linestyle='-', linewidth=1)
    fig_show(fig)

    grid_W = 1000
    grid_H = 600

    maximum_limit = int(grid_W // 2 + grid_W // 2 * (roll_search_window / 2) / 100)
    minimum_limit = int(grid_W // 2 - grid_W // 2 * (roll_search_window / 2) / 100)

    upper_limit = int(grid_W // 2 + grid_W // 2 * (1 / 2) / 100)
    lower_limit = int(grid_W // 2 - grid_W // 2 * (1 / 2) / 100)
    step_window = upper_limit - lower_limit

    input_signal = downsampler(input_signal, speed_up)
    input_signal_period = input_signal_period // speed_up

    mean_q_factor = []

    for window in tqdm(range(minimum_limit, maximum_limit, 4), desc='Find Ideal Roll'):

        grid = np.zeros((grid_H, grid_W), dtype=np.int32)

        ys = []
        for i in range(0, len(input_signal) // (2 * input_signal_period)):
            ys.append(input_signal[int(i * input_signal_period * 2): int((i + 1) * input_signal_period * 2)])

        t_d = np.round(np.linspace(0, grid_W, input_signal_period * 2 + 1, dtype=np.int32))

        y_min = np.nanmin(ys)
        y_max = np.nanmax(ys)
        ys_d = []
        for y in ys:
            ys_d.append(np.round(linear_scale(y, y_min, y_max, 0, grid_H)))

        for y_d in ys_d:
            bres_curve_count(t_d, y_d, grid)

        eye_count = np.zeros(grid_H)

        upper_limit = window + step_window
        lower_limit = window
        eye_range = upper_limit - lower_limit

        for i in range(grid_H):
            for j in range(lower_limit, upper_limit):
                eye_count[i] += grid[i][j]
            eye_count[i] /= eye_range

        eye_center = []
        for i in range(len(eye_count)):
            limit = int(round(eye_count[i], 2) * 100)
            for j in range(limit):
                eye_center.append(i)

        eye_q_factor = np.array(eye_center).reshape(-1, 1)

        gmm = GaussianMixture(n_components=input_signal_levels)
        gmm.fit(eye_q_factor)

        levels = gmm.means_.flatten()
        covariances = np.sqrt(gmm.covariances_.flatten())

        sorted_indices = np.argsort(levels)
        levels = levels[sorted_indices]
        covariances = covariances[sorted_indices]

        q_factor = []
        for i in range(input_signal_levels - 1):
            value = round((levels[i + 1] - levels[i]) / (covariances[i + 1] + covariances[i]), 2)
            q_factor.append(value)

        mean_q_factor.append(round(np.mean(q_factor), 2))

    print(f'Ideal Roll factor: {np.argmax(mean_q_factor)} (MAX Mean Q factor: {mean_q_factor[np.argmax(mean_q_factor)]})')

    return np.argmax(mean_q_factor)
