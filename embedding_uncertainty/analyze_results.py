# analyze_results.py



import torch

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

import glob

import torch.nn.functional as F

import numpy as np

def analyze_results(results_dir="embedding_results"):

    # 1. 加载所有结果文件

    search_path = os.path.join(results_dir, "*.pt")

    result_files = glob.glob(search_path)

    if not result_files:

        print(f"No result files found in '{results_dir}'. Please run 'run_embedding_test.py' first.")

        return



    data = []

    for f in result_files:

        res = torch.load(f)

        # 创建一个易于阅读的配置标签

        det_str = "det" if res['deterministic'] else "nondet"

        res['config_label'] = f"{res['precision']}-{det_str}-run{res['run_id']}"

        data.append(res)

    

    # 转换为Pandas DataFrame

    df = pd.DataFrame(data)

    df = df.sort_values(by=['precision', 'deterministic', 'run_id']).reset_index(drop=True)



    # 2. 定义基准 (Baseline)

    # 我们选择最严格、最可信的配置作为基准

    try:

        baseline_row = df[(df['precision'] == 'fp32') & (df['deterministic'] == True)].iloc[0]

        baseline_embedding = baseline_row['embedding']

        baseline_label = baseline_row['config_label']

        print(f"Using baseline for comparison: {baseline_label}\n")

    except IndexError:

        print("Baseline (fp32, deterministic) not found. Cannot perform comparison.")

        print("Available configs:")

        print(df['config_label'].tolist())

        return



    # 3. 计算与基准的偏差

    l2_distances = []

    cos_similarities = []

    max_abs_diffs = []



    for i, row in df.iterrows():

        current_embedding = row['embedding']

        l2_dist = torch.linalg.norm(baseline_embedding - current_embedding).item()

        cos_sim = F.cosine_similarity(baseline_embedding.unsqueeze(0), current_embedding.unsqueeze(0)).item()

        max_abs_diff = torch.max(torch.abs(baseline_embedding - current_embedding)).item()

        

        l2_distances.append(l2_dist)

        cos_similarities.append(cos_sim)

        max_abs_diffs.append(max_abs_diff)



    df['l2_dist_vs_baseline'] = l2_distances

    df['cos_sim_vs_baseline'] = cos_similarities

    df['max_abs_diff_vs_baseline'] = max_abs_diffs



    print("--- Analysis Results ---")

    print(df[['config_label', 'duration', 'l2_dist_vs_baseline', 'cos_sim_vs_baseline', 'max_abs_diff_vs_baseline']].to_string())

    print("-" * 20)



    # 4. 可视化

    # 图1: L2距离与执行时间对比

    fig, ax1 = plt.subplots(figsize=(16, 8))

    sns.barplot(x='config_label', y='l2_dist_vs_baseline', data=df, ax=ax1, palette='viridis', alpha=0.7)

    ax1.set_ylabel('L2 Distance from Baseline', color='b')

    ax1.set_xlabel('Configuration')

    ax1.tick_params(axis='y', labelcolor='b')

    plt.xticks(rotation=45, ha='right')



    ax2 = ax1.twinx()

    sns.lineplot(x='config_label', y='duration', data=df, ax=ax2, color='r', marker='o', sort=False)

    ax2.set_ylabel('Execution Time (seconds)', color='r')

    ax2.tick_params(axis='y', labelcolor='r')



    plt.title('Embedding Deviation (vs FP32-Det) and Performance')

    fig.tight_layout()

    plt.savefig(os.path.join(results_dir, "deviation_vs_performance.png"))

    print("Saved plot: deviation_vs_performance.png")

    plt.close()



    # 图2: 所有配置对之间的L2距离热力图

    num_configs = len(df)

    dist_matrix = np.zeros((num_configs, num_configs))

    for i in range(num_configs):

        for j in range(num_configs):

            emb_i = df.iloc[i]['embedding']

            emb_j = df.iloc[j]['embedding']

            dist_matrix[i, j] = torch.linalg.norm(emb_i - emb_j).item()



    plt.figure(figsize=(12, 10))

    sns.heatmap(dist_matrix, xticklabels=df['config_label'], yticklabels=df['config_label'], annot=True, fmt=".2e", cmap="magma_r")

    plt.title("Pairwise L2 Distance Matrix Between All Embeddings")

    plt.xticks(rotation=45, ha='right')

    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, "pairwise_distance_heatmap.png"))

    print("Saved plot: pairwise_distance_heatmap.png")

    plt.close()



if __name__ == "__main__":

    analyze_results()
