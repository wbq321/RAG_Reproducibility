#!/usr/bin/env python
"""
generate_cluster_report.py - Generate comprehensive report from cluster experiments
Creates visualizations and detailed analysis of distributed RAG reproducibility
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ClusterReportGenerator:
    """Generate comprehensive reports from distributed test results"""
    
    def __init__(self, job_dir):
        self.job_dir = Path(job_dir)
        self.results_file = self.job_dir / "distributed_results.json"
        self.cluster_info_file = self.job_dir / "cluster_info.json"
        
        # Load results
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        # Load cluster info if available
        if self.cluster_info_file.exists():
            with open(self.cluster_info_file, 'r') as f:
                self.cluster_info = json.load(f)
        else:
            self.cluster_info = None
    
    def generate_full_report(self):
        """Generate all report components"""
        print(f"Generating report for job: {self.job_dir}")
        
        # Create report directory
        report_dir = self.job_dir / "report"
        report_dir.mkdir(exist_ok=True)
        
        # Generate different report sections
        self._generate_scaling_analysis(report_dir)
        self._generate_reproducibility_analysis(report_dir)
        self._generate_performance_analysis(report_dir)
        self._generate_html_report(report_dir)
        
        print(f"Report generated in: {report_dir}")
    
    def _generate_scaling_analysis(self, report_dir):
        """Analyze and visualize scaling behavior"""
        if "scaling" not in self.results:
            return
        
        scaling_data = self.results["scaling"]
        
        # Extract data for plotting
        nodes = []
        qps = []
        jaccard = []
        exact_match = []
        index_time = []
        search_time = []
        
        for config, data in sorted(scaling_data.items()):
            nodes.append(data["num_nodes"])
            qps.append(data["throughput_qps"])
            jaccard.append(data["metrics"]["overlap"]["mean_jaccard"])
            exact_match.append(data["metrics"]["exact_match"]["exact_match_rate"])
            index_time.append(data["avg_index_time"])
            search_time.append(data["avg_search_time"])
        
        # Create scaling plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Throughput Scaling",
                "Reproducibility vs Scale",
                "Time Breakdown",
                "Scaling Efficiency"
            )
        )
        
        # 1. Throughput scaling
        fig.add_trace(
            go.Scatter(x=nodes, y=qps, mode='lines+markers', name='Actual QPS'),
            row=1, col=1
        )
        # Add ideal scaling line
        ideal_qps = [qps[0] * n for n in nodes]
        fig.add_trace(
            go.Scatter(x=nodes, y=ideal_qps, mode='lines', name='Ideal Scaling',
                      line=dict(dash='dash')),
            row=1, col=1
        )
        
        # 2. Reproducibility metrics
        fig.add_trace(
            go.Scatter(x=nodes, y=jaccard, mode='lines+markers', name='Jaccard'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=nodes, y=exact_match, mode='lines+markers', name='Exact Match'),
            row=1, col=2
        )
        
        # 3. Time breakdown
        fig.add_trace(
            go.Bar(x=nodes, y=index_time, name='Index Time'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=nodes, y=search_time, name='Search Time'),
            row=2, col=1
        )
        
        # 4. Scaling efficiency
        efficiency = [q / ideal for q, ideal in zip(qps, ideal_qps)]
        fig.add_trace(
            go.Scatter(x=nodes, y=efficiency, mode='lines+markers',
                      name='Scaling Efficiency'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Number of Nodes")
        fig.update_yaxes(title_text="Queries/Second", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        fig.update_yaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Efficiency", row=2, col=2)
        
        fig.update_layout(
            title="Distributed Scaling Analysis",
            height=800,
            showlegend=True
        )
        
        fig.write_html(report_dir / "scaling_analysis.html")
        
        # Also create matplotlib version for PDF reports
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(nodes, qps, 'o-', label='Actual')
        plt.plot(nodes, ideal_qps, '--', label='Ideal')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Queries/Second')
        plt.title('Throughput Scaling')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(nodes, jaccard, 'o-', label='Jaccard')
        plt.plot(nodes, exact_match, 's-', label='Exact Match')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Score')
        plt.title('Reproducibility vs Scale')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        x = np.arange(len(nodes))
        width = 0.35
        plt.bar(x - width/2, index_time, width, label='Index Time')
        plt.bar(x + width/2, search_time, width, label='Search Time')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Time (seconds)')
        plt.title('Time Breakdown')
        plt.xticks(x, nodes)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(nodes, efficiency, 'o-')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Scaling')
        plt.axhline(y=0.8, color='g', linestyle='--', label='80% Efficiency')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Scaling Efficiency')
        plt.title('Scaling Efficiency')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(report_dir / "scaling_analysis.png", dpi=300)
        plt.close()
    
    def _generate_reproducibility_analysis(self, report_dir):
        """Analyze reproducibility across different configurations"""
        
        # Collect all reproducibility metrics
        reproducibility_data = []
        
        # From scaling experiments
        if "scaling" in self.results:
            for config, data in self.results["scaling"].items():
                metrics = data["metrics"]
                reproducibility_data.append({
                    "experiment": "scaling",
                    "config": config,
                    "nodes": data["num_nodes"],
                    "jaccard": metrics["overlap"]["mean_jaccard"],
                    "jaccard_std": metrics["overlap"]["std_jaccard"],
                    "exact_match": metrics["exact_match"]["exact_match_rate"],
                    "kendall_tau": metrics["rank_correlation"]["mean_kendall_tau"],
                    "score_variance": metrics["score_stability"]["mean_score_variance"]
                })
        
        # From sharding experiments
        if "sharding" in self.results:
            for method, data in self.results["sharding"].items():
                metrics = data["metrics"]
                reproducibility_data.append({
                    "experiment": "sharding",
                    "config": method,
                    "method": method,
                    "jaccard": metrics["overlap"]["mean_jaccard"],
                    "jaccard_std": metrics["overlap"]["std_jaccard"],
                    "exact_match": metrics["exact_match"]["exact_match_rate"],
                    "kendall_tau": metrics["rank_correlation"]["mean_kendall_tau"],
                    "score_variance": metrics["score_stability"]["mean_score_variance"]
                })
        
        df = pd.DataFrame(reproducibility_data)
        
        # Create comprehensive reproducibility plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Jaccard similarity by experiment type
        ax = axes[0, 0]
        if len(df[df['experiment'] == 'scaling']) > 0:
            scaling_df = df[df['experiment'] == 'scaling']
            ax.errorbar(scaling_df['nodes'], scaling_df['jaccard'], 
                       yerr=scaling_df['jaccard_std'], fmt='o-', label='Scaling')
        
        if len(df[df['experiment'] == 'sharding']) > 0:
            sharding_df = df[df['experiment'] == 'sharding']
            x_pos = np.arange(len(sharding_df))
            ax.bar(x_pos, sharding_df['jaccard'], alpha=0.6, label='Sharding')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sharding_df['method'], rotation=45)
        
        ax.set_ylabel('Jaccard Similarity')
        ax.set_title('Reproducibility by Configuration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Correlation between metrics
        ax = axes[0, 1]
        if len(df) > 0:
            ax.scatter(df['jaccard'], df['kendall_tau'], alpha=0.6)
            ax.set_xlabel('Jaccard Similarity')
            ax.set_ylabel('Kendall Tau')
            ax.set_title('Jaccard vs Rank Correlation')
            
            # Add correlation line
            if len(df) > 2:
                z = np.polyfit(df['jaccard'], df['kendall_tau'], 1)
                p = np.poly1d(z)
                ax.plot(df['jaccard'], p(df['jaccard']), "r--", alpha=0.8)
        
        # 3. Score stability
        ax = axes[1, 0]
        if len(df) > 0:
            ax.scatter(df['jaccard'], np.log10(df['score_variance'] + 1e-10), alpha=0.6)
            ax.set_xlabel('Jaccard Similarity')
            ax.set_ylabel('Log10(Score Variance)')
            ax.set_title('Reproducibility vs Score Stability')
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "Summary Statistics\n" + "="*30 + "\n"
        if len(df) > 0:
            summary_text += f"Mean Jaccard: {df['jaccard'].mean():.3f} ± {df['jaccard'].std():.3f}\n"
            summary_text += f"Mean Exact Match: {df['exact_match'].mean():.3f}\n"
            summary_text += f"Mean Kendall Tau: {df['kendall_tau'].mean():.3f}\n"
            summary_text += f"\nBest Configuration:\n"
            best_idx = df['jaccard'].idxmax()
            summary_text += f"  Config: {df.loc[best_idx, 'config']}\n"
            summary_text += f"  Jaccard: {df.loc[best_idx, 'jaccard']:.3f}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontfamily='monospace', fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(report_dir / "reproducibility_analysis.png", dpi=300)
        plt.close()
        
        # Save detailed data
        if len(df) > 0:
            df.to_csv(report_dir / "reproducibility_metrics.csv", index=False)
    
    def _generate_performance_analysis(self, report_dir):
        """Analyze performance characteristics"""
        
        # Create performance dashboard
        fig = go.Figure()
        
        if "scaling" in self.results:
            scaling_data = self.results["scaling"]
            
            # Calculate various performance metrics
            perf_data = []
            for config, data in scaling_data.items():
                n_nodes = data["num_nodes"]
                n_gpus = n_nodes * 2  # Assuming 2 GPUs per node
                qps = data["throughput_qps"]
                
                perf_data.append({
                    "nodes": n_nodes,
                    "gpus": n_gpus,
                    "qps": qps,
                    "qps_per_node": qps / n_nodes,
                    "qps_per_gpu": qps / n_gpus,
                    "index_time": data["avg_index_time"],
                    "search_time": data["avg_search_time"]
                })
            
            perf_df = pd.DataFrame(perf_data)
            
            # Create performance heatmap
            plt.figure(figsize=(10, 8))
            
            # Normalize metrics for heatmap
            metrics = ['qps', 'qps_per_node', 'qps_per_gpu', 'index_time', 'search_time']
            normalized_data = []
            
            for _, row in perf_df.iterrows():
                norm_row = []
                for metric in metrics:
                    if metric in ['index_time', 'search_time']:
                        # For time metrics, lower is better
                        norm_val = 1.0 / (row[metric] + 0.1)
                    else:
                        # For throughput metrics, higher is better
                        norm_val = row[metric]
                    norm_row.append(norm_val)
                normalized_data.append(norm_row)
            
            # Normalize to 0-1 range
            normalized_data = np.array(normalized_data)
            for i in range(normalized_data.shape[1]):
                col_min = normalized_data[:, i].min()
                col_max = normalized_data[:, i].max()
                if col_max > col_min:
                    normalized_data[:, i] = (normalized_data[:, i] - col_min) / (col_max - col_min)
            
            sns.heatmap(normalized_data.T, 
                       xticklabels=[f"{n} nodes" for n in perf_df['nodes']],
                       yticklabels=['QPS', 'QPS/Node', 'QPS/GPU', '1/Index Time', '1/Search Time'],
                       cmap='RdYlGn', annot=True, fmt='.2f')
            
            plt.title('Performance Metrics Heatmap (Normalized)')
            plt.tight_layout()
            plt.savefig(report_dir / "performance_heatmap.png", dpi=300)
            plt.close()
            
            # Save performance data
            perf_df.to_csv(report_dir / "performance_metrics.csv", index=False)
    
    def _generate_html_report(self, report_dir):
        """Generate comprehensive HTML report"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Distributed RAG Reproducibility Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .warning {{
            background-color: #f39c12;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .success {{
            background-color: #27ae60;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Distributed RAG Reproducibility Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Job Directory: {self.job_dir}</p>
    </div>
"""
        
        # Cluster Information Section
        if self.cluster_info:
            html_content += """
    <div class="section">
        <h2>Cluster Configuration</h2>
        <table>
            <tr>
                <th>Node</th>
                <th>Hostname</th>
                <th>CPUs</th>
                <th>Memory (GB)</th>
                <th>GPUs</th>
            </tr>
"""
            for node in self.cluster_info:
                gpu_info = ", ".join([g['name'] for g in node['gpu_info']]) if node['gpu_info'] else "N/A"
                html_content += f"""
            <tr>
                <td>{node['rank']}</td>
                <td>{node['hostname']}</td>
                <td>{node['cpu_count']}</td>
                <td>{node['memory_gb']:.1f}</td>
                <td>{gpu_info}</td>
            </tr>
"""
            html_content += """
        </table>
    </div>
"""
        
        # Key Metrics Section
        html_content += """
    <div class="section">
        <h2>Key Performance Metrics</h2>
"""
        
        if "scaling" in self.results:
            # Find best performing configuration
            best_qps = 0
            best_config = None
            
            for config, data in self.results["scaling"].items():
                if data["throughput_qps"] > best_qps:
                    best_qps = data["throughput_qps"]
                    best_config = data
            
            if best_config:
                html_content += f"""
        <div class="metric">
            <div class="metric-value">{best_qps:.0f}</div>
            <div class="metric-label">Peak QPS</div>
        </div>
        <div class="metric">
            <div class="metric-value">{best_config['num_nodes']}</div>
            <div class="metric-label">Optimal Nodes</div>
        </div>
"""
        
        # Reproducibility Summary
        if "sharding" in self.results:
            best_jaccard = 0
            best_method = None
            
            for method, data in self.results["sharding"].items():
                jaccard = data["metrics"]["overlap"]["mean_jaccard"]
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_method = method
            
            html_content += f"""
        <div class="metric">
            <div class="metric-value">{best_jaccard:.3f}</div>
            <div class="metric-label">Best Jaccard Score</div>
        </div>
        <div class="metric">
            <div class="metric-value">{best_method}</div>
            <div class="metric-label">Best Shard Method</div>
        </div>
"""
        
        html_content += """
    </div>
    
    <div class="section">
        <h2>Detailed Results</h2>
        <h3>Scaling Analysis</h3>
        <img src="scaling_analysis.png" style="max-width: 100%;">
        
        <h3>Reproducibility Analysis</h3>
        <img src="reproducibility_analysis.png" style="max-width: 100%;">
        
        <h3>Performance Heatmap</h3>
        <img src="performance_heatmap.png" style="max-width: 100%;">
    </div>
"""
        
        # Recommendations Section
        html_content += """
    <div class="section">
        <h2>Recommendations</h2>
"""
        
        # Analyze results and provide recommendations
        recommendations = self._generate_recommendations()
        
        for rec in recommendations:
            if rec['type'] == 'warning':
                html_content += f'<div class="warning">{rec["message"]}</div>'
            else:
                html_content += f'<div class="success">{rec["message"]}</div>'
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(report_dir / "report.html", 'w') as f:
            f.write(html_content)
    
    def _generate_recommendations(self):
        """Generate recommendations based on results"""
        recommendations = []
        
        if "scaling" in self.results:
            # Check scaling efficiency
            scaling_data = self.results["scaling"]
            nodes = []
            qps = []
            
            for config, data in scaling_data.items():
                nodes.append(data["num_nodes"])
                qps.append(data["throughput_qps"])
            
            if len(nodes) > 1:
                # Calculate scaling efficiency
                ideal_scaling = [qps[0] * (n / nodes[0]) for n in nodes]
                actual_scaling = qps
                
                efficiency = [a / i for a, i in zip(actual_scaling, ideal_scaling)]
                avg_efficiency = np.mean(efficiency[1:])  # Exclude single node
                
                if avg_efficiency > 0.8:
                    recommendations.append({
                        "type": "success",
                        "message": f"Excellent scaling efficiency: {avg_efficiency:.1%} of ideal"
                    })
                elif avg_efficiency > 0.6:
                    recommendations.append({
                        "type": "warning",
                        "message": f"Moderate scaling efficiency: {avg_efficiency:.1%}. Consider optimizing data distribution."
                    })
                else:
                    recommendations.append({
                        "type": "warning",
                        "message": f"Poor scaling efficiency: {avg_efficiency:.1%}. Review parallelization strategy."
                    })
        
        if "sharding" in self.results:
            # Check reproducibility
            jaccard_scores = []
            for method, data in self.results["sharding"].items():
                jaccard_scores.append(data["metrics"]["overlap"]["mean_jaccard"])
            
            avg_jaccard = np.mean(jaccard_scores)
            
            if avg_jaccard > 0.95:
                recommendations.append({
                    "type": "success",
                    "message": f"Excellent reproducibility: {avg_jaccard:.3f} average Jaccard similarity"
                })
            elif avg_jaccard > 0.85:
                recommendations.append({
                    "type": "warning",
                    "message": f"Good reproducibility: {avg_jaccard:.3f} Jaccard. Consider deterministic mode for improvement."
                })
            else:
                recommendations.append({
                    "type": "warning",
                    "message": f"Low reproducibility: {avg_jaccard:.3f} Jaccard. Enable deterministic operations."
                })
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Generate cluster test report")
    parser.add_argument("--job-dir", type=str, required=True, help="Job output directory")
    
    args = parser.parse_args()
    
    generator = ClusterReportGenerator(args.job_dir)
    generator.generate_full_report()


if __name__ == "__main__":
    main()
