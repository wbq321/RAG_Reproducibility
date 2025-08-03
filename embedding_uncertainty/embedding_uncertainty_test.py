#!/usr/bin/env python3

"""

RAG Query Embedding Uncertainty Testing Suite

用于测试sentence-transformers在不同配置下的embedding稳定性

支持在计算集群上运行，可配置不同的硬件环境

"""



import os

import json

import time

import argparse

import numpy as np

import pandas as pd

from datetime import datetime

from pathlib import Path

from typing import Dict, List, Tuple, Optional

import torch

from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.metrics.pairwise import cosine_similarity

import logging

from tqdm import tqdm

import hashlib



# 配置日志

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

)

logger = logging.getLogger(__name__)



class EmbeddingUncertaintyTester:

    """Query Embedding不确定性测试器"""

    

    def __init__(self, output_dir: str = "./results"):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []

        self.device_info = self._get_device_info()

        

    def _get_device_info(self) -> Dict:

        """获取当前硬件环境信息"""

        info = {

            "cpu_count": os.cpu_count(),

            "cuda_available": torch.cuda.is_available(),

            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,

        }

        

        if torch.cuda.is_available():

            for i in range(torch.cuda.device_count()):

                info[f"gpu_{i}_name"] = torch.cuda.get_device_name(i)

                info[f"gpu_{i}_memory"] = torch.cuda.get_device_properties(i).total_memory

                

        return info

    

    def prepare_test_queries(self) -> Dict[str, List[str]]:

        """准备测试查询文本"""

        queries = {

            "short": [

                "What is the role of CRISPR in gene editing?",

                "量子纠缠的基本原理是什么？",

                "How does deep learning differ from machine learning?",

                "蛋白质折叠的主要驱动力有哪些？",

                "What are the applications of graphene in electronics?"

            ],

            "medium": [

                "Explain the mechanism of protein folding and how misfolding leads to neurodegenerative diseases like Alzheimer's. Include the role of chaperones.",

                "描述机器学习中的过拟合现象，以及常用的正则化方法如L1、L2正则化和Dropout的工作原理。",

                "Discuss the challenges in developing quantum computers and the current approaches to overcome quantum decoherence.",

                "分析CRISPR-Cas9系统的工作机制，包括PAM序列识别、DNA切割和修复机制。",

                "How do transformer models achieve better performance than RNNs in natural language processing tasks?"

            ],

            "long": [

                """Recent advances in single-cell RNA sequencing (scRNA-seq) have revolutionized our understanding of cellular heterogeneity in complex tissues. 

                This technology allows researchers to profile gene expression at the individual cell level, revealing previously unknown cell types and states. 

                The computational challenges include dealing with high-dimensional sparse data, batch effects, and trajectory inference. 

                Machine learning methods such as variational autoencoders and graph neural networks are being developed to address these challenges. 

                What are the key computational methods for analyzing scRNA-seq data and how do they handle the unique characteristics of this data type?""",

                

                """量子计算和经典计算的根本区别在于信息处理的基本单位。经典计算使用比特（0或1），而量子计算使用量子比特（qubit），可以同时处于0和1的叠加态。

                这种量子叠加特性，结合量子纠缠现象，使得量子计算机在某些特定问题上具有指数级的加速潜力。

                然而，量子退相干是实现大规模量子计算的主要障碍。目前的量子纠错码和拓扑量子计算是两种主要的解决方案。

                请详细分析量子计算在密码学、药物发现和材料科学中的潜在应用，以及实现这些应用需要克服的技术挑战。"""

            ]

        }

        return queries

    

    def test_single_configuration(self, 

                                  model_name: str,

                                  query: str,

                                  config: Dict,

                                  num_runs: int = 10) -> Dict:

        """测试单个配置下的embedding稳定性"""

        embeddings = []

        computation_times = []

        

        # 设置设备

        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        

        logger.info(f"Testing configuration: {config}")

        

        for run_idx in range(num_runs):

            try:

                # 设置随机种子

                if 'random_seed' in config and config['random_seed'] is not None:

                    torch.manual_seed(config['random_seed'] + run_idx)

                    np.random.seed(config['random_seed'] + run_idx)

                    if torch.cuda.is_available():

                        torch.cuda.manual_seed_all(config['random_seed'] + run_idx)

                

                # 设置确定性算法（如果需要）

                if config.get('deterministic', False):

                    torch.use_deterministic_algorithms(True)

                    torch.backends.cudnn.deterministic = True

                    torch.backends.cudnn.benchmark = False

                

                # 初始化模型

                start_time = time.time()

                

                model_kwargs = {

                    'device': device

                }

                

                # 添加额外的模型参数

                if 'model_kwargs' in config:

                    model_kwargs.update(config['model_kwargs'])

                

                model = SentenceTransformer(model_name, **model_kwargs)

                

                # 设置数值精度

                if 'dtype' in config:

                    if config['dtype'] == 'float16':

                        model = model.half()

                    elif config['dtype'] == 'bfloat16' and torch.cuda.is_available():

                        model = model.to(torch.bfloat16)

                

                # 生成embedding

                with torch.no_grad():

                    if config.get('use_amp', False) and torch.cuda.is_available():

                        with torch.cuda.amp.autocast():

                            embedding = model.encode(

                                query,

                                batch_size=config.get('batch_size', 32),

                                normalize_embeddings=config.get('normalize_embeddings', True),

                                convert_to_numpy=True,

                                show_progress_bar=False

                            )

                    else:

                        embedding = model.encode(

                            query,

                            batch_size=config.get('batch_size', 32),

                            normalize_embeddings=config.get('normalize_embeddings', True),

                            convert_to_numpy=True,

                            show_progress_bar=False

                        )

                

                computation_time = time.time() - start_time

                embeddings.append(embedding)

                computation_times.append(computation_time)

                

                # 清理模型以释放内存

                del model

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

                    

            except Exception as e:

                logger.error(f"Error in run {run_idx}: {str(e)}")

                continue

        

        if len(embeddings) == 0:

            return None

        

        # 计算统计量

        embeddings = np.array(embeddings)

        mean_embedding = np.mean(embeddings, axis=0)

        std_embedding = np.std(embeddings, axis=0)

        

        # 计算额外的统计指标

        results = {

            'model_name': model_name,

            'query': query[:50] + '...' if len(query) > 50 else query,

            'query_length': len(query.split()),

            'config': config,

            'num_runs': len(embeddings),

            'embedding_dim': embeddings.shape[1],

            'mean_computation_time': np.mean(computation_times),

            'std_computation_time': np.std(computation_times),

            

            # 数值稳定性指标

            'mean_std': np.mean(std_embedding),

            'max_std': np.max(std_embedding),

            'min_std': np.min(std_embedding),

            'mean_cv': np.mean(std_embedding / (np.abs(mean_embedding) + 1e-8)),

            

            # 相似度稳定性

            'cosine_similarities': self._compute_pairwise_similarities(embeddings),

            

            # 每个维度的统计

            'dimension_stats': {

                'high_variance_dims': np.where(std_embedding > np.percentile(std_embedding, 95))[0].tolist(),

                'low_variance_dims': np.where(std_embedding < np.percentile(std_embedding, 5))[0].tolist()

            }

        }

        

        return results

    

    def _compute_pairwise_similarities(self, embeddings: np.ndarray) -> Dict:

        """计算嵌入向量之间的成对相似度"""

        n = len(embeddings)

        if n < 2:

            return {}

        

        similarities = []

        for i in range(n):

            for j in range(i + 1, n):

                sim = cosine_similarity(

                    embeddings[i].reshape(1, -1),

                    embeddings[j].reshape(1, -1)

                )[0, 0]

                similarities.append(sim)

        

        return {

            'mean': np.mean(similarities),

            'std': np.std(similarities),

            'min': np.min(similarities),

            'max': np.max(similarities)

        }

    

    def run_comprehensive_test(self, models: List[str], num_runs: int = 10):

        """运行完整的测试套件"""

        queries = self.prepare_test_queries()

        

        # 定义测试配置

        test_configs = self._generate_test_configs()

        

        total_tests = len(models) * sum(len(q) for q in queries.values()) * len(test_configs)

        

        with tqdm(total=total_tests, desc="Running tests") as pbar:

            for model_name in models:

                logger.info(f"Testing model: {model_name}")

                

                for query_type, query_list in queries.items():

                    for query in query_list:

                        for config_name, config in test_configs.items():

                            result = self.test_single_configuration(

                                model_name, query, config, num_runs

                            )

                            

                            if result:

                                result['query_type'] = query_type

                                result['config_name'] = config_name

                                result['timestamp'] = datetime.now().isoformat()

                                result['device_info'] = self.device_info

                                self.results.append(result)

                            

                            pbar.update(1)

        

        # 保存结果

        self._save_results()

        

    def _generate_test_configs(self) -> Dict[str, Dict]:

        """生成测试配置"""

        configs = {

            # 基准配置

            "baseline": {

                "device": "cuda" if torch.cuda.is_available() else "cpu",

                "batch_size": 32,

                "normalize_embeddings": True,

                "random_seed": 42,

                "deterministic": True

            },

            

            # 批大小测试

            "batch_size_1": {

                "device": "cuda" if torch.cuda.is_available() else "cpu",

                "batch_size": 1,

                "normalize_embeddings": True,

                "random_seed": 42,

                "deterministic": True

            },

            "batch_size_128": {

                "device": "cuda" if torch.cuda.is_available() else "cpu",

                "batch_size": 128,

                "normalize_embeddings": True,

                "random_seed": 42,

                "deterministic": True

            },

            

            # 精度测试

            "float16": {

                "device": "cuda" if torch.cuda.is_available() else "cpu",

                "batch_size": 32,

                "normalize_embeddings": True,

                "random_seed": 42,

                "deterministic": True,

                "dtype": "float16"

            },

            

            # 归一化测试

            "no_normalization": {

                "device": "cuda" if torch.cuda.is_available() else "cpu",

                "batch_size": 32,

                "normalize_embeddings": False,

                "random_seed": 42,

                "deterministic": True

            },

            

            # 非确定性测试

            "non_deterministic": {

                "device": "cuda" if torch.cuda.is_available() else "cpu",

                "batch_size": 32,

                "normalize_embeddings": True,

                "random_seed": None,

                "deterministic": False

            },

            

            # CPU测试

            "cpu_only": {

                "device": "cpu",

                "batch_size": 32,

                "normalize_embeddings": True,

                "random_seed": 42,

                "deterministic": True

            },

            

            # 混合精度测试

            "amp_enabled": {

                "device": "cuda" if torch.cuda.is_available() else "cpu",

                "batch_size": 32,

                "normalize_embeddings": True,

                "random_seed": 42,

                "deterministic": True,

                "use_amp": True

            }

        }

        

        # 如果有多个GPU，添加GPU特定测试

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:

            for i in range(torch.cuda.device_count()):

                configs[f"gpu_{i}"] = {

                    "device": f"cuda:{i}",

                    "batch_size": 32,

                    "normalize_embeddings": True,

                    "random_seed": 42,

                    "deterministic": True

                }

        

        return configs

    

    def _save_results(self):

        """保存测试结果"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        

        # 保存原始结果

        results_file = self.output_dir / f"results_{timestamp}.json"

        with open(results_file, 'w', encoding='utf-8') as f:

            json.dump(self.results, f, indent=2, ensure_ascii=False)

        

        # 创建汇总报告

        self._create_summary_report(timestamp)

        

        logger.info(f"Results saved to {results_file}")

    

    def _create_summary_report(self, timestamp: str):

        """创建汇总报告"""

        report_file = self.output_dir / f"summary_report_{timestamp}.txt"

        

        with open(report_file, 'w', encoding='utf-8') as f:

            f.write("=" * 80 + "\n")

            f.write("RAG Query Embedding Uncertainty Test Report\n")

            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            f.write("=" * 80 + "\n\n")

            

            # 设备信息

            f.write("Device Information:\n")

            for key, value in self.device_info.items():

                f.write(f"  {key}: {value}\n")

            f.write("\n")

            

            # 按模型汇总

            models = set(r['model_name'] for r in self.results)

            for model in models:

                f.write(f"\nModel: {model}\n")

                f.write("-" * 40 + "\n")

                

                model_results = [r for r in self.results if r['model_name'] == model]

                

                # 按配置类型分析

                config_names = set(r['config_name'] for r in model_results)

                for config_name in sorted(config_names):

                    config_results = [r for r in model_results if r['config_name'] == config_name]

                    

                    mean_stds = [r['mean_std'] for r in config_results]

                    mean_times = [r['mean_computation_time'] for r in config_results]

                    

                    f.write(f"\n  Configuration: {config_name}\n")

                    f.write(f"    Number of tests: {len(config_results)}\n")

                    f.write(f"    Average std deviation: {np.mean(mean_stds):.6f} (±{np.std(mean_stds):.6f})\n")

                    f.write(f"    Average computation time: {np.mean(mean_times):.3f}s (±{np.std(mean_times):.3f}s)\n")

                    

                    # 相似度统计

                    cos_sims = [r['cosine_similarities']['mean'] for r in config_results if 'mean' in r['cosine_similarities']]

                    if cos_sims:

                        f.write(f"    Average cosine similarity: {np.mean(cos_sims):.6f}\n")

    

    def visualize_results(self):

        """生成可视化图表"""

        if not self.results:

            logger.warning("No results to visualize")

            return

        

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        viz_dir = self.output_dir / "visualizations"

        viz_dir.mkdir(exist_ok=True)

        

        # 1. 不同配置下的标准差分布

        self._plot_std_distribution(viz_dir, timestamp)

        

        # 2. 计算时间对比

        self._plot_computation_times(viz_dir, timestamp)

        

        # 3. 相似度热力图

        self._plot_similarity_heatmap(viz_dir, timestamp)

        

        # 4. 查询长度影响

        self._plot_query_length_impact(viz_dir, timestamp)

    

    def _plot_std_distribution(self, viz_dir: Path, timestamp: str):

        """绘制标准差分布图"""

        plt.figure(figsize=(12, 8))

        

        # 准备数据

        data_for_plot = []

        for result in self.results:

            data_for_plot.append({

                'model': result['model_name'].split('/')[-1],  # 简化模型名

                'config': result['config_name'],

                'mean_std': result['mean_std']

            })

        

        df = pd.DataFrame(data_for_plot)

        

        # 箱线图

        sns.boxplot(data=df, x='config', y='mean_std', hue='model')

        plt.xticks(rotation=45, ha='right')

        plt.ylabel('Mean Standard Deviation')

        plt.xlabel('Configuration')

        plt.title('Embedding Uncertainty by Configuration')

        plt.tight_layout()

        

        plt.savefig(viz_dir / f'std_distribution_{timestamp}.png', dpi=300)

        plt.close()

    

    def _plot_computation_times(self, viz_dir: Path, timestamp: str):

        """绘制计算时间对比图"""

        plt.figure(figsize=(10, 6))

        

        # 准备数据

        data_for_plot = []

        for result in self.results:

            data_for_plot.append({

                'model': result['model_name'].split('/')[-1],

                'config': result['config_name'],

                'time': result['mean_computation_time']

            })

        

        df = pd.DataFrame(data_for_plot)

        

        # 条形图

        pivot_df = df.pivot_table(values='time', index='config', columns='model', aggfunc='mean')

        pivot_df.plot(kind='bar')

        

        plt.ylabel('Computation Time (seconds)')

        plt.xlabel('Configuration')

        plt.title('Average Computation Time by Configuration')

        plt.xticks(rotation=45, ha='right')

        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        

        plt.savefig(viz_dir / f'computation_times_{timestamp}.png', dpi=300)

        plt.close()

    

    def _plot_similarity_heatmap(self, viz_dir: Path, timestamp: str):

        """绘制相似度热力图"""

        # 为每个模型创建热力图

        models = set(r['model_name'] for r in self.results)

        

        for model in models:

            model_results = [r for r in self.results if r['model_name'] == model]

            configs = sorted(set(r['config_name'] for r in model_results))

            

            # 创建相似度矩阵

            sim_matrix = np.zeros((len(configs), len(configs)))

            

            for i, config1 in enumerate(configs):

                for j, config2 in enumerate(configs):

                    if i == j:

                        sim_matrix[i, j] = 1.0

                    else:

                        # 计算两个配置之间的平均相似度

                        config1_results = [r for r in model_results if r['config_name'] == config1]

                        config2_results = [r for r in model_results if r['config_name'] == config2]

                        

                        if config1_results and config2_results:

                            sim1 = np.mean([r['cosine_similarities']['mean'] for r in config1_results 

                                          if 'mean' in r['cosine_similarities']])

                            sim2 = np.mean([r['cosine_similarities']['mean'] for r in config2_results 

                                          if 'mean' in r['cosine_similarities']])

                            sim_matrix[i, j] = (sim1 + sim2) / 2

            

            # 绘制热力图

            plt.figure(figsize=(10, 8))

            sns.heatmap(sim_matrix, 

                       xticklabels=configs, 

                       yticklabels=configs,

                       annot=True, 

                       fmt='.4f',

                       cmap='YlOrRd',

                       vmin=0.99,

                       vmax=1.0)

            

            plt.title(f'Configuration Similarity Matrix - {model.split("/")[-1]}')

            plt.tight_layout()

            

            model_name_safe = model.replace('/', '_')

            plt.savefig(viz_dir / f'similarity_heatmap_{model_name_safe}_{timestamp}.png', dpi=300)

            plt.close()

    

    def _plot_query_length_impact(self, viz_dir: Path, timestamp: str):

        """绘制查询长度对不确定性的影响"""

        plt.figure(figsize=(10, 6))

        

        # 准备数据

        data_for_plot = []

        for result in self.results:

            data_for_plot.append({

                'query_type': result['query_type'],

                'query_length': result['query_length'],

                'mean_std': result['mean_std'],

                'model': result['model_name'].split('/')[-1]

            })

        

        df = pd.DataFrame(data_for_plot)

        

        # 散点图

        models = df['model'].unique()

        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

        

        for i, model in enumerate(models):

            model_df = df[df['model'] == model]

            plt.scatter(model_df['query_length'], 

                       model_df['mean_std'], 

                       label=model, 

                       alpha=0.6,

                       color=colors[i])

        

        plt.xlabel('Query Length (words)')

        plt.ylabel('Mean Standard Deviation')

        plt.title('Query Length vs Embedding Uncertainty')

        plt.legend()

        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        

        plt.savefig(viz_dir / f'query_length_impact_{timestamp}.png', dpi=300)

        plt.close()





def main():

    parser = argparse.ArgumentParser(description='Test embedding uncertainty in RAG systems')

    parser.add_argument('--models', nargs='+', 

                       default=['sentence-transformers/all-MiniLM-L6-v2',

                               'sentence-transformers/all-mpnet-base-v2'],

                       help='List of models to test')

    parser.add_argument('--num-runs', type=int, default=10,

                       help='Number of runs per configuration')

    parser.add_argument('--output-dir', type=str, default='./uncertainty_results',

                       help='Output directory for results')

    parser.add_argument('--visualize', action='store_true',

                       help='Generate visualization plots')

    

    # 集群相关参数

    parser.add_argument('--gpu-id', type=int, default=None,

                       help='Specific GPU to use')

    parser.add_argument('--num-threads', type=int, default=None,

                       help='Number of CPU threads to use')

    

    args = parser.parse_args()

    

    # 设置环境变量

    if args.gpu_id is not None:

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    

    if args.num_threads is not None:

        torch.set_num_threads(args.num_threads)

        os.environ['OMP_NUM_THREADS'] = str(args.num_threads)

    

    # 创建测试器并运行测试

    tester = EmbeddingUncertaintyTester(output_dir=args.output_dir)

    

    logger.info("Starting embedding uncertainty tests...")

    logger.info(f"Models to test: {args.models}")

    logger.info(f"Number of runs per config: {args.num_runs}")

    

    # 运行测试

    tester.run_comprehensive_test(models=args.models, num_runs=args.num_runs)

    

    # 生成可视化

    if args.visualize:

        logger.info("Generating visualizations...")

        tester.visualize_results()

    

    logger.info("Testing completed!")

    logger.info(f"Results saved to: {args.output_dir}")





if __name__ == "__main__":

    main()
