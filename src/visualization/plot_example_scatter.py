# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/10 22:18
@Auth ： Hongwei
@File ：plot_example_scatter.py
@IDE ：PyCharm
"""
from sklearn.cluster import KMeans
from definitions import *

base_list = [2, 10, 0.3, 0.1]


def log(base, x):
    return np.log(x) / np.log(base)


def plot_traditional_single_mapping(fig, subplot_idx):
    ax = fig.add_subplot(1, 3, subplot_idx)
    points = []
    for idx in range(4):
        random.seed(idx)
        x = np.random.uniform(0.1, 9 / 2 * np.pi, 500)
        y = log(base_list[idx], x) + np.random.uniform(-0.5, 0.5, 500)
        ax.scatter(x, y, color='black', alpha=0.6)
        X = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        points.extend(X.tolist())
    points = np.array(points)
    mapping = np.polyfit(points[:, 0], points[:, 1], 4)
    p = np.poly1d(mapping)
    sub_x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 100)
    ax.plot(sub_x, p(sub_x), linewidth=12, alpha=0.8, color='black')

    ax.set_xlim([0,  4.8 * np.pi])
    ax.set_ylim([-3.5, 5])
    ax.axhline(-3.5, linewidth=5, color='black')
    ax.axvline(linewidth=5, color='black')
    ax.set_xlabel('x\n(a) Traditional single model \nfitting result', fontsize=32)
    ax.set_ylabel('f(x)', fontsize=30, rotation=90)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)


def plot_traditional_sampling_ensemble_mapping(fig, subplot_idx):
    ax = fig.add_subplot(1, 3, subplot_idx)
    color_list = ['#2FAFB5', '#9DE7DA', '#F8B9C4']
    points = []
    for idx in range(4):
        random.seed(idx)
        x = np.random.uniform(0.1, 9 / 2 * np.pi, 500)
        y = log(base_list[idx], x) + np.random.uniform(-0.5, 0.5, 500)
        X = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        points.extend(X.tolist())
    points = np.array(points)
    K = 3
    cluster_model = KMeans(n_clusters=K, random_state=0)
    labels = cluster_model.fit_predict(points)

    for i in range(K):
        class_member_mask = (labels == i)
        xy = points[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=color_list[i], alpha=0.5)
        mapping = np.polyfit(xy[:, 0], xy[:, 1], 4)
        p = np.poly1d(mapping)
        sub_x = np.linspace(np.min(xy[:, 0]), np.max(xy[:, 0]), 100)
        ax.plot(sub_x, p(sub_x), linewidth=12, alpha=1, color=color_list[i])

    ax.set_xlim([0,  4.8 * np.pi])
    ax.set_ylim([-3.5, 5])
    ax.axhline(-3.5, linewidth=5, color='black')
    ax.axvline(linewidth=5, color='black')
    ax.set_xlabel('x\n(b) Traditional sampling or clustering \nfitting result', fontsize=32)
    ax.set_ylabel('f(x)', fontsize=30, rotation=90)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)


def plot_shap_clustering_mappings(fig, col_num, subplot_idx):
    ax = fig.add_subplot(1, col_num, subplot_idx)
    color_list = ['#2FAFB5', '#9DE7DA', '#F8B9C4', '#ee8a5f']
    for idx in range(4):
        random.seed(idx)
        x = np.linspace(0.15, 9 / 2 * np.pi, 500)
        y = log(base_list[idx], x)
        x_noise = np.random.uniform(0.1, 9 / 2 * np.pi, 500)
        y_noise = log(base_list[idx], x_noise) + np.random.uniform(-0.5, 0.5, 500)
        ax.plot(x, y, linewidth=10, alpha=0.8, color=color_list[idx])
        ax.scatter(x_noise, y_noise, color=color_list[idx])
    ax.set_xlim([0,  4.8 * np.pi])
    ax.set_ylim([-3.5, 5])
    ax.axhline(-3.5, linewidth=5, color='black')
    ax.axvline(linewidth=5, color='black')
    ax.set_xlabel('x\n(c) Mapping relationship-based \nclustering fitting result', fontsize=32)
    ax.set_ylabel('f(x)', fontsize=30, rotation=90)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)


if __name__ == '__main__':
    fig = plt.figure(figsize=(24, 8), dpi=200)
    plot_traditional_single_mapping(fig, subplot_idx=1)
    plot_traditional_sampling_ensemble_mapping(fig, subplot_idx=2)
    plot_shap_clustering_mappings(fig, col_num=3, subplot_idx=3)
    plt.tight_layout()
    plt.savefig(FIG_DIR + '/illustration_example.pdf', transparent=True)
    plt.show()
