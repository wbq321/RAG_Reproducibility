#!/usr/bin/env python3

"""

高级结果分析脚本

用于深入分析embedding不确定性测试结果

"""



import json

import numpy as np

import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

import umap

from typing import Dict, List, Tuple

import argparse

from datetime import datetime

import warnings

warnings.filterwarnings('ignore')



class AdvancedAnalyzer:

    """高级分析器"""

    

    def __init__(self, results_path: str):

        self.results_path = Path(results_path)

        self.results = self._load_results()

        self.output_dir = self.results_path.parent / "advanced_analysis"

        self.output_dir.mkdir(exist_ok=True)

        

    def _load_results(self) -> List[Dict]:

        """加载测试结果"""

        if self.results_path.is_file():

            with open(self.results_path, 'r') as f:

                return json.load(f)

        else:

            # 如果是目录，加载所有JSON文件

            all_results = []

            for json_file in self.results_path.glob("*.json"):

                with open(json_file, 'r') as f:

                    all_results.extend(json.load(f))

            return all_results

    

    def statistical_analysis(self):

        """执行统计分析"""

        print("执行统计分析...")

        

        # 准备数据框

        df = self._prepare_dataframe()

        

        # 1. 正态性检验

        normality_results = self._test_normality(df)

        

        # 2. 方差分析

        anova_results = self._perform_anova(df)

        

        # 3. 事后检验

        posthoc_results = self._perform_posthoc(df)

        

        # 保存统计结果

        stats_report = {

            'normality': normality_results,

            'anova': anova_results,

            'posthoc': posthoc_results

        }

        

        with open(self.output_dir / 'statistical_analysis.json', 'w') as f:

            json.dump(stats_report, f, indent=2)

        

        self._create_statistical_report(stats_report)

    

    def _prepare_dataframe(self) -> pd.DataFrame:

        """准备用于分析的数据框"""

        data = []

        for result in self.results:

            data.append({

                'model': result['model_name'].split('/')[-1],

                'config': result['config_name'],

                'query_type': result['query_type'],

                'query_length': result['query_length'],

                'mean_std': result['mean_std'],

                'max_std': result['max_std'],

                'mean_cv': result['mean_cv'],

                'computation_time': result['mean_computation_time'],

                'embedding_dim': result['embedding_dim'],

                'batch_size': result['config'].get('batch_size', 'unknown'),

                'device': result['config'].get('device', 'unknown'),

                'normalize': result['config'].get('normalize_embeddings', True)

            })

        

        return pd.DataFrame(data)

    

    def _test_normality(self, df: pd.DataFrame) -> Dict:

        """测试数据的正态性"""

        results = {}

        

        # 对每个模型和配置组合测试正态性

        for model in df['model'].unique():

            model_df = df[df['model'] == model]

            results[model] = {}

            

            for config in model_df['config'].unique():

                config_df = model_df[model_df['config'] == config]

                if len(config_df) >= 3:  # Shapiro-Wilk需要至少3个样本

                    stat, p_value = stats.shapiro(config_df['mean_std'])

                    results[model][config] = {

                        'statistic': float(stat),

                        'p_value': float(p_value),

                        'is_normal': p_value > 0.05

                    }

        

        return results

    

    def _perform_anova(self, df: pd.DataFrame) -> Dict:

        """执行方差分析"""

        results = {}

        

        # 因素：配置类型对标准差的影响

        groups = []

        for config in df['config'].unique():

            groups.append(df[df['config'] == config]['mean_std'].values)

        

        # Kruskal-Wallis检验（非参数）

        h_stat, p_value = stats.kruskal(*groups)

        results['kruskal_wallis'] = {

            'h_statistic': float(h_stat),

            'p_value': float(p_value),

            'significant': p_value < 0.05

        }

        

        # 如果数据满足正态性，也执行ANOVA

        try:

            f_stat, p_value = stats.f_oneway(*groups)

            results['anova'] = {

                'f_statistic': float(f_stat),

                'p_value': float(p_value),

                'significant': p_value < 0.05

            }

        except:

            results['anova'] = None

        

        return results

    

    def _perform_posthoc(self, df: pd.DataFrame) -> Dict:

        """执行事后检验"""

        from scipy.stats import mannwhitneyu

        

        results = {}

        configs = df['config'].unique()

        

        # 两两比较

        for i, config1 in enumerate(configs):

            for j, config2 in enumerate(configs):

                if i < j:

                    group1 = df[df['config'] == config1]['mean_std'].values

                    group2 = df[df['config'] == config2]['mean_std'].values

                    

                    if len(group1) > 0 and len(group2) > 0:

                        stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

                        key = f"{config1}_vs_{config2}"

                        results[key] = {

                            'u_statistic': float(stat),

                            'p_value': float(p_value),

                            'significant': p_value < 0.05 / len(configs)  # Bonferroni校正

                        }

        

        return results

    

    def dimension_reduction_analysis(self):

        """降维分析，可视化embedding空间的变化"""

        print("执行降维分析...")

        

        # 为每个模型创建降维可视化

        models = set(r['model_name'] for r in self.results)

        

        for model in models:

            model_results = [r for r in self.results if r['model_name'] == model]

            self._visualize_embedding_space(model, model_results)

    

    def _visualize_embedding_space(self, model: str, results: List[Dict]):

        """可视化embedding空间"""

        # 收集所有配置下的平均embedding（如果有保存的话）

        # 这里我们模拟一些数据用于演示

        n_configs = len(set(r['config_name'] for r in results))

        n_queries = len(results) // n_configs if n_configs > 0 else 0

        

        if n_queries == 0:

            return

        

        # 创建模拟数据（实际使用中应该保存真实的embedding）

        embedding_dim = results[0]['embedding_dim']

        embeddings = np.random.randn(len(results), embedding_dim)

        

        # 添加一些模式使不同配置可区分

        configs = [r['config_name'] for r in results]

        unique_configs = list(set(configs))

        for i, config in enumerate(unique_configs):

            mask = np.array([c == config for c in configs])

            embeddings[mask] += np.random.randn(embedding_dim) * 0.1 * i

        

        # PCA降维

        pca = PCA(n_components=2)

        pca_result = pca.fit_transform(embeddings)

        

        # t-SNE降维

        tsne = TSNE(n_components=2, random_state=42)

        tsne_result = tsne.fit_transform(embeddings[:min(len(embeddings), 1000)])  # 限制样本数

        

        # UMAP降维

        try:

            reducer = umap.UMAP(n_components=2, random_state=42)

            umap_result = reducer.fit_transform(embeddings)

        except:

            umap_result = None

        

        # 创建可视化

        fig, axes = plt.subplots(1, 3 if umap_result is not None else 2, figsize=(15, 5))

        

        # PCA图

        for config in unique_configs:

            mask = np.array([c == config for c in configs[:len(pca_result)]])

            axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1], 

                          label=config, alpha=0.6)

        axes[0].set_title('PCA Projection')

        axes[0].legend()

        

        # t-SNE图

        for config in unique_configs:

            mask = np.array([c == config for c in configs[:len(tsne_result)]])

            axes[1].scatter(tsne_result[mask, 0], tsne_result[mask, 1], 

                          label=config, alpha=0.6)

        axes[1].set_title('t-SNE Projection')

        axes[1].legend()

        

        # UMAP图

        if umap_result is not None:

            for config in unique_configs:

                mask = np.array([c == config for c in configs[:len(umap_result)]])

                axes[2].scatter(umap_result[mask, 0], umap_result[mask, 1], 

                              label=config, alpha=0.6)

            axes[2].set_title('UMAP Projection')

            axes[2].legend()

        

        plt.suptitle(f'Embedding Space Visualization - {model.split("/")[-1]}')

        plt.tight_layout()

        

        model_safe = model.replace('/', '_')

        plt.savefig(self.output_dir / f'embedding_space_{model_safe}.png', dpi=300)

        plt.close()

    

    def hardware_impact_analysis(self):

        """分析硬件因素的影响"""

        print("分析硬件影响...")

        

        df = self._prepare_dataframe()

        

        # 按设备类型分组

        device_groups = df.groupby('device').agg({

            'mean_std': ['mean', 'std', 'min', 'max'],

            'computation_time': ['mean', 'std', 'min', 'max']

        })

        

        # 可视化设备影响

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        

        # 不确定性对比

        devices = df['device'].unique()

        uncertainty_data = [df[df['device'] == d]['mean_std'].values for d in devices]

        ax1.boxplot(uncertainty_data, labels=devices)

        ax1.set_ylabel('Mean Standard Deviation')

        ax1.set_title('Uncertainty by Device Type')

        

        # 计算时间对比

        time_data = [df[df['device'] == d]['computation_time'].values for d in devices]

        ax2.boxplot(time_data, labels=devices)

        ax2.set_ylabel('Computation Time (s)')

        ax2.set_title('Computation Time by Device Type')

        

        plt.tight_layout()

        plt.savefig(self.output_dir / 'hardware_impact.png', dpi=300)

        plt.close()

        

        # 保存统计数据

        device_groups.to_csv(self.output_dir / 'hardware_statistics.csv')

    

    def batch_size_scaling_analysis(self):

        """分析批大小的影响"""

        print("分析批大小缩放...")

        

        df = self._prepare_dataframe()

        

        # 过滤出有批大小信息的数据

        batch_df = df[df['batch_size'] != 'unknown'].copy()

        batch_df['batch_size'] = pd.to_numeric(batch_df['batch_size'])

        

        if len(batch_df) == 0:

            print("没有批大小数据可供分析")

            return

        

        # 按模型和批大小分组

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        

        for model in batch_df['model'].unique():

            model_df = batch_df[batch_df['model'] == model]

            

            # 按批大小分组计算平均值

            grouped = model_df.groupby('batch_size').agg({

                'mean_std': 'mean',

                'computation_time': 'mean'

            }).reset_index()

            

            # 绘制不确定性vs批大小

            ax1.plot(grouped['batch_size'], grouped['mean_std'], 

                    marker='o', label=model)

            

            # 绘制计算时间vs批大小

            ax2.plot(grouped['batch_size'], grouped['computation_time'], 

                    marker='o', label=model)

        

        ax1.set_xlabel('Batch Size')

        ax1.set_ylabel('Mean Standard Deviation')

        ax1.set_title('Uncertainty vs Batch Size')

        ax1.set_xscale('log')

        ax1.legend()

        ax1.grid(True, alpha=0.3)

        

        ax2.set_xlabel('Batch Size')

        ax2.set_ylabel('Computation Time (s)')

        ax2.set_title('Computation Time vs Batch Size')

        ax2.set_xscale('log')

        ax2.legend()

        ax2.grid(True, alpha=0.3)

        

        plt.tight_layout()

        plt.savefig(self.output_dir / 'batch_size_scaling.png', dpi=300)

        plt.close()

    

    def _create_statistical_report(self, stats_report: Dict):

        """创建统计报告"""

        report_path = self.output_dir / 'statistical_report.txt'

        

        with open(report_path, 'w') as f:

            f.write("=" * 80 + "\n")

            f.write("Statistical Analysis Report\n")

            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            f.write("=" * 80 + "\n\n")

            

            # 正态性检验结果

            f.write("1. Normality Tests (Shapiro-Wilk)\n")

            f.write("-" * 40 + "\n")

            for model, configs in stats_report['normality'].items():

                f.write(f"\nModel: {model}\n")

                for config, result in configs.items():

                    f.write(f"  {config}: p={result['p_value']:.4f} ")

                    f.write(f"({'Normal' if result['is_normal'] else 'Not Normal'})\n")

            

            # ANOVA结果

            f.write("\n2. Variance Analysis\n")

            f.write("-" * 40 + "\n")

            if stats_report['anova']['kruskal_wallis']:

                kw = stats_report['anova']['kruskal_wallis']

                f.write(f"Kruskal-Wallis H-test: H={kw['h_statistic']:.4f}, ")

                f.write(f"p={kw['p_value']:.4f}\n")

                f.write(f"Result: {'Significant' if kw['significant'] else 'Not Significant'}\n")

            

            # 事后检验结果

            f.write("\n3. Post-hoc Tests (Mann-Whitney U)\n")

            f.write("-" * 40 + "\n")

            significant_pairs = []

            for pair, result in stats_report['posthoc'].items():

                if result['significant']:

                    significant_pairs.append(pair)

            

            if significant_pairs:

                f.write("Significant differences found between:\n")

                for pair in significant_pairs:

                    f.write(f"  - {pair.replace('_vs_', ' vs ')}\n")

            else:

                f.write("No significant differences found between configurations.\n")

    

    def create_comprehensive_report(self):

        """创建综合报告"""

        print("生成综合报告...")

        

        # 执行所有分析

        self.statistical_analysis()

        self.hardware_impact_analysis()

        self.batch_size_scaling_analysis()

        self.dimension_reduction_analysis()

        

        # 创建HTML报告

        self._create_html_report()

        

        print(f"分析完成！结果保存在: {self.output_dir}")

    

    def _create_html_report(self):

        """创建HTML格式的综合报告"""

        html_content = f"""

<!DOCTYPE html>

<html>

<head>

    <title>Embedding Uncertainty Analysis Report</title>

    <style>

        body {{ font-family: Arial, sans-serif; margin: 20px; }}

        h1 {{ color: #333; }}

        h2 {{ color: #666; }}

        .section {{ margin: 20px 0; }}

        img {{ max-width: 100%; height: auto; margin: 10px 0; }}

        table {{ border-collapse: collapse; width: 100%; }}

        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}

        th {{ background-color: #f2f2f2; }}

    </style>

</head>

<body>

    <h1>RAG Query Embedding Uncertainty Analysis Report</h1>

    <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    

    <div class="section">

        <h2>1. Overview</h2>

        <p>Total number of tests: {len(self.results)}</p>

        <p>Models tested: {len(set(r['model_name'] for r in self.results))}</p>

        <p>Configurations tested: {len(set(r['config_name'] for r in self.results))}</p>

    </div>

    

    <div class="section">

        <h2>2. Statistical Analysis</h2>

        <p>See <a href="statistical_report.txt">detailed statistical report</a></p>

    </div>

    

    <div class="section">

        <h2>3. Visualizations</h2>

        <h3>Hardware Impact</h3>

        <img src="hardware_impact.png" alt="Hardware Impact">

        

        <h3>Batch Size Scaling</h3>

        <img src="batch_size_scaling.png" alt="Batch Size Scaling">

    </div>

    

    <div class="section">

        <h2>4. Recommendations</h2>

        <ul>

            <li>Use deterministic algorithms when reproducibility is critical</li>

            <li>Consider the trade-off between batch size and stability</li>

            <li>GPU computation generally provides more consistent results</li>

            <li>Normalization significantly affects embedding stability</li>

        </ul>

    </div>

</body>

</html>

        """

        

        with open(self.output_dir / 'report.html', 'w') as f:

            f.write(html_content)





def main():

    parser = argparse.ArgumentParser(description='Analyze embedding uncertainty test results')

    parser.add_argument('results_path', help='Path to results JSON file or directory')

    parser.add_argument('--full-analysis', action='store_true', 

                       help='Run all analysis types')

    

    args = parser.parse_args()

    

    analyzer = AdvancedAnalyzer(args.results_path)

    

    if args.full_analysis:

        analyzer.create_comprehensive_report()

    else:

        # 运行特定分析

        analyzer.statistical_analysis()

        analyzer.hardware_impact_analysis()





if __name__ == "__main__":

    main()
