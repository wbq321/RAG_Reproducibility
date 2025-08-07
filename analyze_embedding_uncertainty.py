#!/usr/bin/env python3
"""
Embedding Uncertainty Analysis Script
Analyzes embedding reproducibility and precision comparison results
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np

# Plotting imports (optional - will work without if not installed)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Plotting libraries not available. Install matplotlib, seaborn, pandas for visualizations.")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class EmbeddingUncertaintyAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.analyze_dir = self.results_dir / "analyze"
        self.analyze_dir.mkdir(exist_ok=True)

        self.reproducibility_data = None
        self.precision_data = None

    def load_embedding_reproducibility_results(self):
        """Load embedding reproducibility results"""
        repro_file = self.results_dir / "embedding_reproducibility_results.json"
        if repro_file.exists():
            try:
                with open(repro_file, 'r', encoding='utf-8') as f:
                    self.reproducibility_data = json.load(f)
                print(f"✅ Loaded embedding reproducibility results")
                return True
            except Exception as e:
                print(f"❌ Error loading embedding reproducibility results: {e}")
                return False
        else:
            print(f"❌ Embedding reproducibility results file not found: {repro_file}")
            return False

    def parse_precision_comparison_report(self):
        """Parse the precision comparison markdown report"""
        report_file = self.results_dir / "precision_comparison_report.md"
        if not report_file.exists():
            print(f"❌ Precision comparison report not found: {report_file}")
            return False

        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract metadata
            metadata = {}
            metadata_pattern = r'\*\*(Generated|Model|Device|Number of texts)\*\*:\s*(.+)'
            for match in re.finditer(metadata_pattern, content):
                key = match.group(1).lower().replace(' ', '_')
                value = match.group(2).strip()
                metadata[key] = value

            # Extract precision configurations
            config_pattern = r'- \*\*(.+?)\*\*'
            configs = re.findall(config_pattern, content)

            # Extract summary results
            summary = {}
            summary_pattern = r'- \*\*(Most similar|Least similar|Average L2|Average cosine).+?\*\*:\s*(.+)'
            for match in re.finditer(summary_pattern, content):
                key = match.group(1).lower().replace(' ', '_').replace('average_', 'avg_')
                value = match.group(2).strip()
                summary[key] = value

            # Extract detailed comparisons
            comparisons = {}

            # Pattern to match each comparison section
            comparison_sections = re.split(r'### (.+?)\n', content)[1:]  # Skip first empty element

            for i in range(0, len(comparison_sections), 2):
                if i + 1 < len(comparison_sections):
                    comparison_name = comparison_sections[i].strip()
                    comparison_content = comparison_sections[i + 1]

                    # Extract metrics from this comparison
                    comparison_data = {}

                    # L2 Distance metrics
                    l2_pattern = r'\*\*L2 Distance:\*\*\n- Mean: (.+?)\n- Std: (.+?)\n(?:.*?\n)*?- 95th percentile: (.+?)\n'
                    l2_match = re.search(l2_pattern, comparison_content, re.DOTALL)
                    if l2_match:
                        comparison_data['l2_distance'] = {
                            'mean': l2_match.group(1).strip(),
                            'std': l2_match.group(2).strip(),
                            'percentile_95': l2_match.group(3).strip()
                        }

                    # Cosine Similarity metrics
                    cos_pattern = r'\*\*Cosine Similarity:\*\*\n- Mean: (.+?)\n- Min: (.+?)\n'
                    cos_match = re.search(cos_pattern, comparison_content)
                    if cos_match:
                        comparison_data['cosine_similarity'] = {
                            'mean': cos_match.group(1).strip(),
                            'min': cos_match.group(2).strip()
                        }

                    # Element-wise differences
                    elem_pattern = r'\*\*Element-wise Differences:\*\*\n- Mean absolute difference: (.+?)\n- Max absolute difference: (.+?)\n- Fraction of small differences \(<1e-6\): (.+?)\n'
                    elem_match = re.search(elem_pattern, comparison_content)
                    if elem_match:
                        comparison_data['element_wise_diff'] = {
                            'mean_abs_diff': elem_match.group(1).strip(),
                            'max_abs_diff': elem_match.group(2).strip(),
                            'fraction_small_diff': elem_match.group(3).strip()
                        }

                    comparisons[comparison_name] = comparison_data

            self.precision_data = {
                'metadata': metadata,
                'configurations': configs,
                'summary': summary,
                'comparisons': comparisons
            }

            print(f"✅ Parsed precision comparison report ({len(comparisons)} comparisons)")
            return True

        except Exception as e:
            print(f"❌ Error parsing precision comparison report: {e}")
            return False

    def analyze_reproducibility_stability(self) -> Dict[str, Any]:
        """Analyze embedding reproducibility stability"""
        if not self.reproducibility_data:
            return {}

        analysis = {
            'configurations_tested': list(self.reproducibility_data.keys()),
            'stability_summary': {},
            'deterministic_vs_nondeterministic': {},
            'performance_metrics': {}
        }

        # Analyze each configuration
        for config_name, config_data in self.reproducibility_data.items():
            metrics = config_data.get('metrics', {})
            config_info = config_data.get('config', {})

            # Extract key stability metrics
            l2_mean = float(metrics.get('l2_distance', {}).get('mean', 0))
            cosine_mean = float(metrics.get('cosine_similarity', {}).get('mean', 1))
            exact_match_rate = float(metrics.get('exact_match_rate', 0))

            # Extract timing information
            timing = metrics.get('timing', {})
            mean_duration = timing.get('mean_duration', 0)

            analysis['stability_summary'][config_name] = {
                'precision': config_info.get('precision'),
                'deterministic': config_info.get('deterministic'),
                'l2_distance_mean': l2_mean,
                'cosine_similarity_mean': cosine_mean,
                'exact_match_rate': exact_match_rate,
                'mean_encoding_time': mean_duration,
                'stability_grade': 'Perfect' if exact_match_rate == 1.0 else 'Variable'
            }

        # Compare deterministic vs non-deterministic
        det_configs = {k: v for k, v in analysis['stability_summary'].items()
                      if 'Deterministic' in k}
        nondet_configs = {k: v for k, v in analysis['stability_summary'].items()
                         if 'Non-Deterministic' in k}

        if det_configs and nondet_configs:
            det_l2_mean = np.mean([config['l2_distance_mean'] for config in det_configs.values()])
            nondet_l2_mean = np.mean([config['l2_distance_mean'] for config in nondet_configs.values()])

            det_exact_match = np.mean([config['exact_match_rate'] for config in det_configs.values()])
            nondet_exact_match = np.mean([config['exact_match_rate'] for config in nondet_configs.values()])

            analysis['deterministic_vs_nondeterministic'] = {
                'deterministic': {
                    'avg_l2_distance': det_l2_mean,
                    'avg_exact_match_rate': det_exact_match,
                    'num_configs': len(det_configs)
                },
                'non_deterministic': {
                    'avg_l2_distance': nondet_l2_mean,
                    'avg_exact_match_rate': nondet_exact_match,
                    'num_configs': len(nondet_configs)
                },
                'difference_detected': abs(det_l2_mean - nondet_l2_mean) > 1e-10
            }

        return analysis

    def analyze_precision_effects(self) -> Dict[str, Any]:
        """Analyze precision-related effects from comparison data"""
        if not self.precision_data:
            return {}

        comparisons = self.precision_data.get('comparisons', {})

        analysis = {
            'cross_precision_comparisons': {},
            'deterministic_mode_comparisons': {},
            'precision_ranking': {},
            'insights': []
        }

        # Categorize comparisons
        cross_precision = {}
        det_vs_nondet = {}

        for comp_name, comp_data in comparisons.items():
            # Check if this is a within-precision deterministic vs non-deterministic comparison
            # These should have the same precision type on both sides (e.g., "FP32 DETERMINISTIC VS FP32 NONDETERMINISTIC")
            if 'DETERMINISTIC VS' in comp_name and 'NONDETERMINISTIC' in comp_name:
                parts = comp_name.split(' VS ')
                if len(parts) == 2:
                    left_precision = parts[0].split()[0]  # e.g., "FP32" from "FP32 DETERMINISTIC"
                    right_precision = parts[1].split()[0]  # e.g., "FP32" from "FP32 NONDETERMINISTIC"
                    if left_precision == right_precision:
                        # Same precision type, deterministic vs non-deterministic
                        det_vs_nondet[comp_name] = comp_data
                    else:
                        # Different precision types, this is cross-precision
                        cross_precision[comp_name] = comp_data
                else:
                    # Fallback: if we can't parse properly, treat as det vs nondet
                    det_vs_nondet[comp_name] = comp_data
            elif ('DETERMINISTIC VS' in comp_name and 'DETERMINISTIC' in comp_name.split(' VS ')[1]) or \
                 ('NONDETERMINISTIC VS' in comp_name and 'NONDETERMINISTIC' in comp_name.split(' VS ')[1]):
                # Cross-precision comparisons (same deterministic mode)
                cross_precision[comp_name] = comp_data

        analysis['cross_precision_comparisons'] = cross_precision
        analysis['deterministic_mode_comparisons'] = det_vs_nondet

        # Analyze precision effects
        precision_l2_distances = {}
        for comp_name, comp_data in cross_precision.items():
            l2_data = comp_data.get('l2_distance', {})
            l2_mean_str = l2_data.get('mean', '0')

            try:
                l2_mean = float(l2_mean_str.replace('e-', 'E-').replace('e+', 'E+'))
                precision_l2_distances[comp_name] = l2_mean
            except (ValueError, TypeError):
                continue

        # Rank precisions by similarity (lower L2 distance = more similar)
        if precision_l2_distances:
            sorted_precisions = sorted(precision_l2_distances.items(), key=lambda x: x[1])
            analysis['precision_ranking'] = {
                'most_similar_pair': sorted_precisions[0][0] if sorted_precisions else None,
                'least_similar_pair': sorted_precisions[-1][0] if sorted_precisions else None,
                'all_rankings': sorted_precisions
            }

        # Analyze deterministic vs non-deterministic differences
        det_nondet_differences = {}
        for comp_name, comp_data in det_vs_nondet.items():
            l2_data = comp_data.get('l2_distance', {})
            l2_mean_str = l2_data.get('mean', '0')

            try:
                l2_mean = float(l2_mean_str.replace('e-', 'E-').replace('e+', 'E+'))
                det_nondet_differences[comp_name] = l2_mean
            except (ValueError, TypeError):
                continue

        # Generate insights
        insights = []

        # Check if deterministic vs non-deterministic shows differences
        zero_diff_count = sum(1 for diff in det_nondet_differences.values() if diff == 0)
        if zero_diff_count > 0:
            insights.append(f"⚠️ {zero_diff_count}/{len(det_nondet_differences)} precision types show no difference between deterministic and non-deterministic modes")

        # Check precision stability
        if precision_l2_distances:
            max_diff = max(precision_l2_distances.values())
            min_diff = min(precision_l2_distances.values())
            if max_diff > 1e-2:
                insights.append(f"⚠️ Significant precision differences detected (max L2 distance: {max_diff:.2e})")
            elif max_diff < 1e-5:
                insights.append(f"✅ All precision types show high similarity (max L2 distance: {max_diff:.2e})")

        analysis['insights'] = insights

        return analysis

    def generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive embedding uncertainty analysis"""
        print("🔬 Generating comprehensive embedding uncertainty analysis...")

        analysis = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'embedding_uncertainty',
                'data_sources': []
            },
            'reproducibility_analysis': {},
            'precision_analysis': {},
            'cross_analysis': {},
            'summary_findings': {},
            'recommendations': [],
            'user_questions': {}
        }

        # Track data sources
        if self.reproducibility_data:
            analysis['metadata']['data_sources'].append('embedding_reproducibility_results.json')
        if self.precision_data:
            analysis['metadata']['data_sources'].append('precision_comparison_report.md')

        # Analyze reproducibility
        if self.reproducibility_data:
            analysis['reproducibility_analysis'] = self.analyze_reproducibility_stability()

        # Analyze precision effects
        if self.precision_data:
            analysis['precision_analysis'] = self.analyze_precision_effects()

        # Cross-analysis (if both data sources available)
        if self.reproducibility_data and self.precision_data:
            analysis['cross_analysis'] = self.cross_validate_findings()

        # Answer specific user questions
        analysis['user_questions'] = self.answer_user_questions(analysis)

        # Generate summary findings
        analysis['summary_findings'] = self.generate_summary_findings(analysis)

        # Generate recommendations
        analysis['recommendations'] = self.generate_recommendations(analysis)

        return analysis

    def cross_validate_findings(self) -> Dict[str, Any]:
        """Cross-validate findings between reproducibility and precision analysis"""
        cross_analysis = {
            'consistency_check': {},
            'validation_status': 'unknown'
        }

        # Check if reproducibility results align with precision comparison
        repro_summary = self.reproducibility_data
        precision_det_vs_nondet = self.precision_data.get('comparisons', {})

        # Look for deterministic vs non-deterministic comparisons in precision data
        det_nondet_comps = {k: v for k, v in precision_det_vs_nondet.items()
                           if 'DETERMINISTIC VS' in k and 'NONDETERMINISTIC' in k}

        if det_nondet_comps:
            # Check if precision data shows zero differences for det vs nondet
            zero_diff_count = 0
            total_count = len(det_nondet_comps)

            for comp_name, comp_data in det_nondet_comps.items():
                l2_mean_str = comp_data.get('l2_distance', {}).get('mean', '0')
                try:
                    l2_mean = float(l2_mean_str.replace('e-', 'E-').replace('e+', 'E+'))
                    if l2_mean == 0:
                        zero_diff_count += 1
                except (ValueError, TypeError):
                    continue

            cross_analysis['consistency_check'] = {
                'zero_diff_precision_comparisons': zero_diff_count,
                'total_precision_comparisons': total_count,
                'percentage_zero_diff': (zero_diff_count / total_count * 100) if total_count > 0 else 0
            }

            # Validation status
            if zero_diff_count == total_count:
                cross_analysis['validation_status'] = 'consistent_no_differences'
            elif zero_diff_count == 0:
                cross_analysis['validation_status'] = 'consistent_all_differences'
            else:
                cross_analysis['validation_status'] = 'mixed_results'

        return cross_analysis

    def answer_user_questions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Answer specific user questions about configuration reproducibility and differences"""
        user_questions = {
            'question_1_reproducibility': {
                'question': 'For every config (8 total), is it reproducible?',
                'analysis': {},
                'summary': ''
            },
            'question_2_config_differences': {
                'question': 'For different config (fixed det/non-det, fixed precision), are they different?',
                'analysis': {},
                'summary': ''
            }
        }

        # Question 1: Reproducibility of each configuration
        repro_analysis = analysis.get('reproducibility_analysis', {})
        stability_summary = repro_analysis.get('stability_summary', {})

        config_reproducibility = {}
        reproducible_count = 0
        total_configs = 0

        # Expected 8 configurations: 4 precisions × 2 deterministic modes
        expected_configs = [
            'FP32 Deterministic', 'FP32 Non-Deterministic',
            'FP16 Deterministic', 'FP16 Non-Deterministic',
            'BF16 Deterministic', 'BF16 Non-Deterministic',
            'TF32 Deterministic', 'TF32 Non-Deterministic'
        ]

        for config_name in expected_configs:
            if config_name in stability_summary:
                config_data = stability_summary[config_name]
                exact_match = config_data.get('exact_match_rate', 0)
                l2_distance = config_data.get('l2_distance_mean', float('inf'))

                is_reproducible = exact_match == 1.0 and l2_distance == 0.0
                config_reproducibility[config_name] = {
                    'reproducible': is_reproducible,
                    'exact_match_rate': exact_match,
                    'l2_distance': l2_distance,
                    'status': 'Perfect' if is_reproducible else 'Variable'
                }

                if is_reproducible:
                    reproducible_count += 1
                total_configs += 1
            else:
                config_reproducibility[config_name] = {
                    'reproducible': None,
                    'exact_match_rate': None,
                    'l2_distance': None,
                    'status': 'Not Tested'
                }

        user_questions['question_1_reproducibility']['analysis'] = config_reproducibility

        if total_configs > 0:
            reproducibility_rate = (reproducible_count / total_configs) * 100
            user_questions['question_1_reproducibility']['summary'] = (
                f"{reproducible_count}/{total_configs} configurations are perfectly reproducible "
                f"({reproducibility_rate:.1f}%). "
                f"{len(expected_configs) - total_configs} configurations were not tested."
            )
        else:
            user_questions['question_1_reproducibility']['summary'] = "No reproducibility data available for analysis."

        # Question 2: Differences between configurations
        precision_analysis = analysis.get('precision_analysis', {})
        cross_precision_comps = precision_analysis.get('cross_precision_comparisons', {})
        det_mode_comps = precision_analysis.get('deterministic_mode_comparisons', {})

        config_differences = {
            'deterministic_cross_precision': {},
            'nondeterministic_cross_precision': {},
            'within_precision_det_vs_nondet': {}
        }

        # Analyze cross-precision differences (same deterministic mode)
        det_cross_precision = {}
        nondet_cross_precision = {}

        for comp_name, comp_data in cross_precision_comps.items():
            l2_data = comp_data.get('l2_distance', {})
            l2_mean_str = l2_data.get('mean', '0')

            try:
                l2_mean = float(l2_mean_str.replace('e-', 'E-').replace('e+', 'E+'))

                # Determine if this is deterministic or non-deterministic comparison
                # Check both sides of the comparison
                parts = comp_name.split(' VS ')
                if len(parts) == 2:
                    left_part = parts[0].strip()
                    right_part = parts[1].strip()

                    # Check for non-deterministic first (since NONDETERMINISTIC contains DETERMINISTIC)
                    if 'NONDETERMINISTIC' in left_part and 'NONDETERMINISTIC' in right_part:
                        # Both sides are non-deterministic
                        nondet_cross_precision[comp_name] = {
                            'l2_distance': l2_mean,
                            'different': l2_mean > 1e-10,
                            'comparison_type': 'nondeterministic_cross_precision'
                        }
                    elif 'DETERMINISTIC' in left_part and 'DETERMINISTIC' in right_part and 'NONDETERMINISTIC' not in left_part and 'NONDETERMINISTIC' not in right_part:
                        # Both sides are deterministic (and not non-deterministic)
                        det_cross_precision[comp_name] = {
                            'l2_distance': l2_mean,
                            'different': l2_mean > 1e-10,
                            'comparison_type': 'deterministic_cross_precision'
                        }
            except (ValueError, TypeError):
                continue

        config_differences['deterministic_cross_precision'] = det_cross_precision
        config_differences['nondeterministic_cross_precision'] = nondet_cross_precision

        # Analyze within-precision deterministic vs non-deterministic differences
        det_vs_nondet_diffs = {}
        for comp_name, comp_data in det_mode_comps.items():
            l2_data = comp_data.get('l2_distance', {})
            l2_mean_str = l2_data.get('mean', '0')

            try:
                l2_mean = float(l2_mean_str.replace('e-', 'E-').replace('e+', 'E+'))
                precision_type = comp_name.split(' ')[0]

                det_vs_nondet_diffs[comp_name] = {
                    'precision_type': precision_type,
                    'l2_distance': l2_mean,
                    'different': l2_mean > 1e-10,
                    'comparison_type': 'within_precision_det_vs_nondet'
                }
            except (ValueError, TypeError):
                continue

        config_differences['within_precision_det_vs_nondet'] = det_vs_nondet_diffs

        user_questions['question_2_config_differences']['analysis'] = config_differences

        # Generate summary for question 2
        summary_parts = []

        # Cross-precision differences (deterministic)
        if det_cross_precision:
            det_different_count = sum(1 for comp in det_cross_precision.values() if comp['different'])
            det_total = len(det_cross_precision)
            summary_parts.append(
                f"Deterministic cross-precision: {det_different_count}/{det_total} comparisons show differences"
            )

        # Cross-precision differences (non-deterministic)
        if nondet_cross_precision:
            nondet_different_count = sum(1 for comp in nondet_cross_precision.values() if comp['different'])
            nondet_total = len(nondet_cross_precision)
            summary_parts.append(
                f"Non-deterministic cross-precision: {nondet_different_count}/{nondet_total} comparisons show differences"
            )

        # Within-precision deterministic vs non-deterministic
        if det_vs_nondet_diffs:
            det_nondet_different_count = sum(1 for comp in det_vs_nondet_diffs.values() if comp['different'])
            det_nondet_total = len(det_vs_nondet_diffs)
            summary_parts.append(
                f"Deterministic vs non-deterministic: {det_nondet_different_count}/{det_nondet_total} comparisons show differences"
            )

        if summary_parts:
            user_questions['question_2_config_differences']['summary'] = "; ".join(summary_parts)
        else:
            user_questions['question_2_config_differences']['summary'] = "No configuration difference data available for analysis."

        return user_questions

    def generate_summary_findings(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate key summary findings"""
        findings = {
            'key_findings': [],
            'reproducibility_grade': 'unknown',
            'precision_stability': 'unknown',
            'deterministic_behavior': 'unknown'
        }

        # Reproducibility findings
        repro_analysis = analysis.get('reproducibility_analysis', {})
        if repro_analysis:
            stability_summary = repro_analysis.get('stability_summary', {})
            perfect_configs = sum(1 for config in stability_summary.values()
                                if config.get('stability_grade') == 'Perfect')
            total_configs = len(stability_summary)

            if total_configs > 0:
                perfect_percentage = (perfect_configs / total_configs) * 100
                findings['reproducibility_grade'] = f"{perfect_percentage:.1f}% Perfect"

                if perfect_percentage == 100:
                    findings['key_findings'].append("✅ All tested configurations show perfect reproducibility")
                elif perfect_percentage >= 80:
                    findings['key_findings'].append(f"⚠️ Most configurations ({perfect_percentage:.1f}%) show perfect reproducibility")
                else:
                    findings['key_findings'].append(f"❌ Low reproducibility: only {perfect_percentage:.1f}% perfect")

        # Precision findings
        precision_analysis = analysis.get('precision_analysis', {})
        if precision_analysis:
            insights = precision_analysis.get('insights', [])
            findings['key_findings'].extend(insights)

            # Determine precision stability
            det_vs_nondet = precision_analysis.get('deterministic_mode_comparisons', {})
            zero_diff_count = 0
            for comp_data in det_vs_nondet.values():
                l2_mean_str = comp_data.get('l2_distance', {}).get('mean', '0')
                try:
                    l2_mean = float(l2_mean_str.replace('e-', 'E-').replace('e+', 'E+'))
                    if l2_mean == 0:
                        zero_diff_count += 1
                except (ValueError, TypeError):
                    continue

            if zero_diff_count == len(det_vs_nondet) and len(det_vs_nondet) > 0:
                findings['deterministic_behavior'] = 'identical'
                findings['key_findings'].append("❗ Deterministic and non-deterministic modes produce identical results")
            elif zero_diff_count == 0 and len(det_vs_nondet) > 0:
                findings['deterministic_behavior'] = 'different'
                findings['key_findings'].append("✅ Deterministic and non-deterministic modes show expected differences")
            else:
                findings['deterministic_behavior'] = 'mixed'

        return findings

    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        summary = analysis.get('summary_findings', {})

        # Reproducibility recommendations
        repro_grade = summary.get('reproducibility_grade', '')
        if 'Perfect' in repro_grade:
            if '100.0%' in repro_grade:
                recommendations.append("✅ Current configuration is optimal for reproducibility")
            else:
                recommendations.append("⚠️ Consider investigating non-perfect configurations for potential improvements")

        # Deterministic behavior recommendations
        det_behavior = summary.get('deterministic_behavior', '')
        if det_behavior == 'identical':
            recommendations.append("🔧 CRITICAL: Fix non-deterministic mode implementation - it should differ from deterministic mode")
            recommendations.append("🔧 Verify random seed reset in non-deterministic mode")
            recommendations.append("🔧 Ensure model state is properly reset between configurations")
        elif det_behavior == 'different':
            recommendations.append("✅ Deterministic controls are working correctly")

        # Precision recommendations
        precision_analysis = analysis.get('precision_analysis', {})
        if precision_analysis:
            ranking = precision_analysis.get('precision_ranking', {})
            most_similar = ranking.get('most_similar_pair', '')
            if 'FP32' in most_similar and 'TF32' in most_similar:
                recommendations.append("💡 FP32 and TF32 show highest similarity - TF32 may be suitable for performance optimization")

            least_similar = ranking.get('least_similar_pair', '')
            if 'BF16' in least_similar:
                recommendations.append("⚠️ BF16 shows largest differences - use with caution in precision-critical applications")

        # General recommendations
        recommendations.extend([
            "📊 Consider running tests with larger sample sizes for statistical significance",
            "🔄 Implement continuous monitoring of embedding reproducibility in production",
            "📝 Document precision requirements for your specific use case"
        ])

        return recommendations

    def save_analysis_results(self, analysis: Dict[str, Any]):
        """Save analysis results to files"""
        # Save JSON results
        json_file = self.analyze_dir / "embedding_uncertainty_analysis.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"💾 Saved JSON analysis: {json_file}")

        # Generate markdown report
        self.generate_markdown_report(analysis)

    def generate_markdown_report(self, analysis: Dict[str, Any]):
        """Generate comprehensive markdown report"""
        report_file = self.analyze_dir / "embedding_uncertainty_analysis_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 🔬 Embedding Uncertainty Analysis Report\n\n")

            # Metadata
            metadata = analysis.get('metadata', {})
            f.write(f"**Generated**: {metadata.get('timestamp', 'Unknown')}\n")
            f.write(f"**Analysis Type**: {metadata.get('analysis_type', 'Unknown')}\n")
            f.write(f"**Data Sources**: {', '.join(metadata.get('data_sources', []))}\n\n")

            # Executive Summary
            f.write("## 📋 Executive Summary\n\n")
            summary = analysis.get('summary_findings', {})

            f.write(f"- **Reproducibility Grade**: {summary.get('reproducibility_grade', 'Unknown')}\n")
            f.write(f"- **Deterministic Behavior**: {summary.get('deterministic_behavior', 'Unknown')}\n")
            f.write(f"- **Precision Stability**: {summary.get('precision_stability', 'Unknown')}\n\n")

            # Key Findings
            key_findings = summary.get('key_findings', [])
            if key_findings:
                f.write("### 🎯 Key Findings\n\n")
                for finding in key_findings:
                    f.write(f"- {finding}\n")
                f.write("\n")

            # User Questions Analysis
            user_questions = analysis.get('user_questions', {})
            if user_questions:
                f.write("## ❓ User Questions Analysis\n\n")

                # Question 1: Configuration Reproducibility
                q1 = user_questions.get('question_1_reproducibility', {})
                if q1:
                    f.write("### Question 1: Configuration Reproducibility\n\n")
                    f.write(f"**Question**: {q1.get('question', 'Unknown')}\n\n")
                    f.write(f"**Summary**: {q1.get('summary', 'No analysis available')}\n\n")

                    q1_analysis = q1.get('analysis', {})
                    if q1_analysis:
                        f.write("**Detailed Results**:\n\n")
                        f.write("| Configuration | Reproducible | Exact Match Rate | L2 Distance | Status |\n")
                        f.write("|---------------|--------------|------------------|-------------|--------|\n")

                        for config_name, config_data in q1_analysis.items():
                            reproducible = config_data.get('reproducible')
                            exact_match = config_data.get('exact_match_rate')
                            l2_distance = config_data.get('l2_distance')
                            status = config_data.get('status', 'Unknown')

                            if reproducible is None:
                                repro_str = "❓ Not Tested"
                                exact_str = "N/A"
                                l2_str = "N/A"
                            else:
                                repro_str = "✅ Yes" if reproducible else "❌ No"
                                exact_str = f"{exact_match:.3f}" if exact_match is not None else "N/A"
                                l2_str = f"{l2_distance:.2e}" if l2_distance is not None else "N/A"

                            f.write(f"| {config_name} | {repro_str} | {exact_str} | {l2_str} | {status} |\n")
                        f.write("\n")

                # Question 2: Configuration Differences
                q2 = user_questions.get('question_2_config_differences', {})
                if q2:
                    f.write("### Question 2: Configuration Differences\n\n")
                    f.write(f"**Question**: {q2.get('question', 'Unknown')}\n\n")
                    f.write(f"**Summary**: {q2.get('summary', 'No analysis available')}\n\n")

                    q2_analysis = q2.get('analysis', {})

                    # Deterministic cross-precision comparisons
                    det_cross = q2_analysis.get('deterministic_cross_precision', {})
                    if det_cross:
                        f.write("**Deterministic Cross-Precision Comparisons**:\n\n")
                        f.write("| Comparison | L2 Distance | Different | Status |\n")
                        f.write("|------------|-------------|-----------|--------|\n")

                        for comp_name, comp_data in det_cross.items():
                            l2_dist = comp_data.get('l2_distance', 0)
                            different = comp_data.get('different', False)
                            status = "✅ Different" if different else "⚠️ Similar"

                            f.write(f"| {comp_name} | {l2_dist:.2e} | {different} | {status} |\n")
                        f.write("\n")

                    # Non-deterministic cross-precision comparisons
                    nondet_cross = q2_analysis.get('nondeterministic_cross_precision', {})
                    if nondet_cross:
                        f.write("**Non-Deterministic Cross-Precision Comparisons**:\n\n")
                        f.write("| Comparison | L2 Distance | Different | Status |\n")
                        f.write("|------------|-------------|-----------|--------|\n")

                        for comp_name, comp_data in nondet_cross.items():
                            l2_dist = comp_data.get('l2_distance', 0)
                            different = comp_data.get('different', False)
                            status = "✅ Different" if different else "⚠️ Similar"

                            f.write(f"| {comp_name} | {l2_dist:.2e} | {different} | {status} |\n")
                        f.write("\n")

                    # Within-precision deterministic vs non-deterministic
                    within_precision = q2_analysis.get('within_precision_det_vs_nondet', {})
                    if within_precision:
                        f.write("**Within-Precision: Deterministic vs Non-Deterministic**:\n\n")
                        f.write("| Precision Type | L2 Distance | Different | Status |\n")
                        f.write("|----------------|-------------|-----------|--------|\n")

                        for comp_name, comp_data in within_precision.items():
                            precision_type = comp_data.get('precision_type', 'Unknown')
                            l2_dist = comp_data.get('l2_distance', 0)
                            different = comp_data.get('different', False)
                            status = "✅ Different" if different else "⚠️ Identical"

                            f.write(f"| {precision_type} | {l2_dist:.2e} | {different} | {status} |\n")
                        f.write("\n")

            # Reproducibility Analysis
            repro_analysis = analysis.get('reproducibility_analysis', {})
            if repro_analysis:
                f.write("## 🔄 Reproducibility Analysis\n\n")

                # Configuration summary
                stability_summary = repro_analysis.get('stability_summary', {})
                if stability_summary:
                    f.write("### Configuration Stability\n\n")
                    f.write("| Configuration | Precision | Deterministic | L2 Distance | Cosine Similarity | Exact Match | Stability |\n")
                    f.write("|---------------|-----------|---------------|-------------|-------------------|-------------|----------|\n")

                    for config_name, config_data in stability_summary.items():
                        precision = config_data.get('precision', 'Unknown')
                        deterministic = config_data.get('deterministic', 'Unknown')
                        l2_dist = f"{config_data.get('l2_distance_mean', 0):.2e}"
                        cos_sim = f"{config_data.get('cosine_similarity_mean', 1):.6f}"
                        exact_match = f"{config_data.get('exact_match_rate', 0):.3f}"
                        stability = config_data.get('stability_grade', 'Unknown')

                        f.write(f"| {config_name} | {precision} | {deterministic} | {l2_dist} | {cos_sim} | {exact_match} | {stability} |\n")
                    f.write("\n")

                # Deterministic vs Non-deterministic
                det_vs_nondet = repro_analysis.get('deterministic_vs_nondeterministic', {})
                if det_vs_nondet:
                    f.write("### Deterministic vs Non-Deterministic Comparison\n\n")
                    det_data = det_vs_nondet.get('deterministic', {})
                    nondet_data = det_vs_nondet.get('non_deterministic', {})

                    f.write(f"**Deterministic Mode**:\n")
                    f.write(f"- Average L2 Distance: {det_data.get('avg_l2_distance', 0):.2e}\n")
                    f.write(f"- Average Exact Match Rate: {det_data.get('avg_exact_match_rate', 0):.3f}\n")
                    f.write(f"- Configurations: {det_data.get('num_configs', 0)}\n\n")

                    f.write(f"**Non-Deterministic Mode**:\n")
                    f.write(f"- Average L2 Distance: {nondet_data.get('avg_l2_distance', 0):.2e}\n")
                    f.write(f"- Average Exact Match Rate: {nondet_data.get('avg_exact_match_rate', 0):.3f}\n")
                    f.write(f"- Configurations: {nondet_data.get('num_configs', 0)}\n\n")

                    difference_detected = det_vs_nondet.get('difference_detected', False)
                    f.write(f"**Difference Detected**: {'Yes' if difference_detected else 'No'}\n\n")

            # Precision Analysis
            precision_analysis = analysis.get('precision_analysis', {})
            if precision_analysis:
                f.write("## 🎯 Precision Analysis\n\n")

                # Precision ranking
                ranking = precision_analysis.get('precision_ranking', {})
                if ranking:
                    f.write("### Precision Similarity Ranking\n\n")
                    f.write(f"**Most Similar Pair**: {ranking.get('most_similar_pair', 'Unknown')}\n")
                    f.write(f"**Least Similar Pair**: {ranking.get('least_similar_pair', 'Unknown')}\n\n")

                    all_rankings = ranking.get('all_rankings', [])
                    if all_rankings:
                        f.write("**All Pairwise Comparisons** (sorted by L2 distance):\n\n")
                        for comp_name, l2_dist in all_rankings:
                            f.write(f"- {comp_name}: {l2_dist:.2e}\n")
                        f.write("\n")

                # Deterministic mode comparisons
                det_mode_comps = precision_analysis.get('deterministic_mode_comparisons', {})
                if det_mode_comps:
                    f.write("### Deterministic vs Non-Deterministic Mode Differences\n\n")
                    f.write("| Precision Type | L2 Distance | Status |\n")
                    f.write("|----------------|-------------|--------|\n")

                    for comp_name, comp_data in det_mode_comps.items():
                        precision_type = comp_name.split(' ')[0]
                        l2_mean_str = comp_data.get('l2_distance', {}).get('mean', '0')
                        try:
                            l2_mean = float(l2_mean_str.replace('e-', 'E-').replace('e+', 'E+'))
                            status = "⚠️ Identical" if l2_mean == 0 else "✅ Different"
                            f.write(f"| {precision_type} | {l2_mean:.2e} | {status} |\n")
                        except (ValueError, TypeError):
                            f.write(f"| {precision_type} | {l2_mean_str} | ❓ Unknown |\n")
                    f.write("\n")

            # Cross Analysis
            cross_analysis = analysis.get('cross_analysis', {})
            if cross_analysis:
                f.write("## 🔗 Cross-Validation Analysis\n\n")

                consistency = cross_analysis.get('consistency_check', {})
                if consistency:
                    zero_diff = consistency.get('zero_diff_precision_comparisons', 0)
                    total = consistency.get('total_precision_comparisons', 0)
                    percentage = consistency.get('percentage_zero_diff', 0)

                    f.write(f"**Consistency Check**: {zero_diff}/{total} comparisons show zero differences ({percentage:.1f}%)\n")
                    f.write(f"**Validation Status**: {cross_analysis.get('validation_status', 'Unknown')}\n\n")

            # Recommendations
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                f.write("## 💡 Recommendations\n\n")
                for rec in recommendations:
                    f.write(f"- {rec}\n")
                f.write("\n")

            f.write("---\n")
            f.write("*Report generated by EmbeddingUncertaintyAnalyzer*\n")

        print(f"📄 Generated report: {report_file}")

    def generate_precision_heatmaps(self, analysis: Dict[str, Any]):
        """Generate separate heatmaps for deterministic and non-deterministic cross-precision comparisons"""
        if not PLOTTING_AVAILABLE:
            print("⚠️ Skipping heatmap generation - plotting libraries not available")
            return

        try:
            print("🎨 Generating precision comparison heatmaps...")

            # Extract precision analysis data
            precision_analysis = analysis.get('precision_analysis', {})
            cross_precision = precision_analysis.get('cross_precision_comparisons', {})

            if not cross_precision:
                print("⚠️ No cross-precision comparison data found for heatmaps")
                return

            # Define precision types
            precisions = ['FP32', 'FP16', 'BF16', 'TF32']

            # Extract deterministic and non-deterministic data
            det_data = {}
            nondet_data = {}

            for comp_name, comp_data in cross_precision.items():
                l2_data = comp_data.get('l2_distance', {})
                l2_mean_str = l2_data.get('mean', '0')

                try:
                    # Handle both string and float types
                    if isinstance(l2_mean_str, str):
                        l2_mean = float(l2_mean_str.replace('e-', 'E-').replace('e+', 'E+'))
                    else:
                        l2_mean = float(l2_mean_str)

                    # Parse comparison name to extract precision types
                    parts = comp_name.split(' VS ')
                    if len(parts) == 2:
                        left_part = parts[0].strip()
                        right_part = parts[1].strip()

                        # Extract precision types
                        left_precision = left_part.split()[0]
                        right_precision = right_part.split()[0]

                        # Determine if deterministic or non-deterministic
                        # Check for non-deterministic first (since NONDETERMINISTIC contains DETERMINISTIC)
                        if 'NONDETERMINISTIC' in left_part and 'NONDETERMINISTIC' in right_part:
                            # Both non-deterministic
                            nondet_data[(left_precision, right_precision)] = l2_mean
                        elif 'DETERMINISTIC' in left_part and 'DETERMINISTIC' in right_part and 'NONDETERMINISTIC' not in left_part and 'NONDETERMINISTIC' not in right_part:
                            # Both deterministic (and not non-deterministic)
                            det_data[(left_precision, right_precision)] = l2_mean

                except (ValueError, TypeError) as e:
                    print(f"⚠️ Error parsing L2 distance for {comp_name}: {e}")
                    continue

            # Create symmetric matrices
            def create_distance_matrix(data, precisions):
                n = len(precisions)
                matrix = np.zeros((n, n))

                # Fill matrix with data
                for (p1, p2), distance in data.items():
                    if p1 in precisions and p2 in precisions:
                        i = precisions.index(p1)
                        j = precisions.index(p2)
                        matrix[i, j] = distance
                        matrix[j, i] = distance  # Make symmetric

                return matrix

            det_matrix = create_distance_matrix(det_data, precisions)
            nondet_matrix = create_distance_matrix(nondet_data, precisions)

            # Check if we have data
            if det_matrix.max() == 0 and nondet_matrix.max() == 0:
                print("⚠️ No valid precision comparison data found for heatmaps")
                return

            # Common color map settings
            vmin = 0
            vmax = max(det_matrix.max(), nondet_matrix.max()) if det_matrix.max() > 0 or nondet_matrix.max() > 0 else 1e-3

            # Create deterministic heatmap
            if det_matrix.max() > 0:
                plt.figure(figsize=(10, 8))

                # Create heatmap with seaborn for better styling
                det_df = pd.DataFrame(det_matrix, index=precisions, columns=precisions)

                # Custom annotations - only show non-zero values
                annot_matrix = np.zeros_like(det_matrix, dtype=object)
                for i in range(len(precisions)):
                    for j in range(len(precisions)):
                        if i == j:
                            annot_matrix[i, j] = '0.00e+00'
                        elif det_matrix[i, j] > 0:
                            annot_matrix[i, j] = f'{det_matrix[i, j]:.2e}'
                        else:
                            annot_matrix[i, j] = ''

                sns.heatmap(det_df, annot=annot_matrix, fmt='', cmap='YlOrRd',
                           square=True, linewidths=0.5, cbar_kws={'label': 'L2 Distance'},
                           annot_kws={'fontsize': 16, 'fontweight': 'bold'},
                           vmin=vmin, vmax=vmax)

                plt.title('Deterministic Cross-Precision Comparisons\n(L2 Distance)',
                         fontsize=22, fontweight='bold', pad=20)
                plt.xlabel('Precision Type', fontsize=22, fontweight='bold')
                plt.ylabel('Precision Type', fontsize=22, fontweight='bold')

                # Make axis tick labels larger and bold
                plt.xticks(fontsize=22, fontweight='bold')
                plt.yticks(fontsize=22, fontweight='bold')

                # Make colorbar labels larger and bold
                cbar = plt.gca().collections[0].colorbar
                cbar.ax.tick_params(labelsize=14)
                cbar.set_label('L2 Distance', fontsize=2, fontweight='bold')

                plt.tight_layout()

                # Save deterministic heatmap
                det_output_path = self.analyze_dir / 'deterministic_precision_heatmap.png'
                plt.savefig(det_output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()

                print(f"📊 Deterministic heatmap saved to: {det_output_path}")

            # Create non-deterministic heatmap
            if nondet_matrix.max() > 0:
                plt.figure(figsize=(10, 8))

                # Create heatmap with seaborn for better styling
                nondet_df = pd.DataFrame(nondet_matrix, index=precisions, columns=precisions)

                # Custom annotations - only show non-zero values
                annot_matrix = np.zeros_like(nondet_matrix, dtype=object)
                for i in range(len(precisions)):
                    for j in range(len(precisions)):
                        if i == j:
                            annot_matrix[i, j] = '0.00e+00'
                        elif nondet_matrix[i, j] > 0:
                            annot_matrix[i, j] = f'{nondet_matrix[i, j]:.2e}'
                        else:
                            annot_matrix[i, j] = ''

                sns.heatmap(nondet_df, annot=annot_matrix, fmt='', cmap='YlOrRd',
                           square=True, linewidths=0.5, cbar_kws={'label': 'L2 Distance'},
                           annot_kws={'fontsize': 16, 'fontweight': 'bold'},
                           vmin=vmin, vmax=vmax)

                plt.title('Non-Deterministic Cross-Precision Comparisons\n(L2 Distance)',
                         fontsize=18, fontweight='bold', pad=20)
                plt.xlabel('Precision Type', fontsize=16, fontweight='bold')
                plt.ylabel('Precision Type', fontsize=16, fontweight='bold')

                # Make axis tick labels larger and bold
                plt.xticks(fontsize=16, fontweight='bold')
                plt.yticks(fontsize=16, fontweight='bold')

                # Make colorbar labels larger and bold
                cbar = plt.gca().collections[0].colorbar
                cbar.ax.tick_params(labelsize=12)
                cbar.set_label('L2 Distance', fontsize=16, fontweight='bold')

                plt.tight_layout()

                # Save non-deterministic heatmap
                nondet_output_path = self.analyze_dir / 'nondeterministic_precision_heatmap.png'
                plt.savefig(nondet_output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()

                print(f"📊 Non-deterministic heatmap saved to: {nondet_output_path}")

            # Create a comparison plot if both exist
            if det_matrix.max() > 0 and nondet_matrix.max() > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

                # Deterministic subplot
                det_df = pd.DataFrame(det_matrix, index=precisions, columns=precisions)
                annot_matrix_det = np.zeros_like(det_matrix, dtype=object)
                for i in range(len(precisions)):
                    for j in range(len(precisions)):
                        if i == j:
                            annot_matrix_det[i, j] = '0.00e+00'
                        elif det_matrix[i, j] > 0:
                            annot_matrix_det[i, j] = f'{det_matrix[i, j]:.2e}'
                        else:
                            annot_matrix_det[i, j] = ''

                sns.heatmap(det_df, annot=annot_matrix_det, fmt='', cmap='YlOrRd',
                           square=True, linewidths=0.5, ax=ax1, cbar=False,
                           annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                           vmin=vmin, vmax=vmax)
                ax1.set_title('Deterministic Mode', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Precision Type', fontsize=12)
                ax1.set_ylabel('Precision Type', fontsize=12)

                # Non-deterministic subplot
                nondet_df = pd.DataFrame(nondet_matrix, index=precisions, columns=precisions)
                annot_matrix_nondet = np.zeros_like(nondet_matrix, dtype=object)
                for i in range(len(precisions)):
                    for j in range(len(precisions)):
                        if i == j:
                            annot_matrix_nondet[i, j] = '0.00e+00'
                        elif nondet_matrix[i, j] > 0:
                            annot_matrix_nondet[i, j] = f'{nondet_matrix[i, j]:.2e}'
                        else:
                            annot_matrix_nondet[i, j] = ''

                im = sns.heatmap(nondet_df, annot=annot_matrix_nondet, fmt='', cmap='YlOrRd',
                               square=True, linewidths=0.5, ax=ax2,
                               annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                               vmin=vmin, vmax=vmax)
                ax2.set_title('Non-Deterministic Mode', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Precision Type', fontsize=12)
                ax2.set_ylabel('Precision Type', fontsize=12)

                # Add shared colorbar
                cbar = plt.colorbar(im.collections[0], ax=[ax1, ax2], shrink=0.8, aspect=30)
                cbar.set_label('L2 Distance', rotation=270, labelpad=20, fontsize=12)

                plt.suptitle('Cross-Precision Comparison: Deterministic vs Non-Deterministic',
                            fontsize=16, fontweight='bold', y=1.02)
                plt.tight_layout()

                # Save comparison plot
                comparison_output_path = self.analyze_dir / 'precision_comparison_side_by_side.png'
                plt.savefig(comparison_output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()

                print(f"📊 Side-by-side comparison saved to: {comparison_output_path}")

                # Check if they are identical
                if np.allclose(det_matrix, nondet_matrix, atol=1e-10):
                    print("⚠️ CRITICAL: Deterministic and non-deterministic modes produce identical results!")
                    print("   This suggests the non-deterministic mode is not working as expected.")

        except Exception as e:
            print(f"❌ Error generating heatmaps: {e}")
            import traceback
            traceback.print_exc()

    def generate_mean_execution_time_plot(self):
        """Generate a simple, focused plot showing only mean execution times"""
        if not PLOTTING_AVAILABLE:
            print("⚠️ Skipping mean execution time plot - plotting libraries not available")
            return

        if not self.reproducibility_data:
            print("⚠️ No reproducibility data available for mean execution time plot")
            return

        try:
            print("⏱️ Generating mean execution time plot...")

            # Extract timing data
            timing_data = {}

            for config_name, config_results in self.reproducibility_data.items():
                if isinstance(config_results, dict) and 'metrics' in config_results:
                    if 'timings' in config_results:
                        raw_timings = config_results['timings']
                        if len(raw_timings) > 1:
                            # Exclude first run (cold start)
                            warm_timings = raw_timings[1:]
                            mean_duration = np.mean(warm_timings)
                            std_duration = np.std(warm_timings)
                            timing_data[config_name] = {
                                'mean': mean_duration,
                                'std': std_duration
                            }
                        else:
                            # Use available data
                            timing_metrics = config_results['metrics'].get('timing', {})
                            if timing_metrics:
                                timing_data[config_name] = {
                                    'mean': timing_metrics.get('mean_duration', 0),
                                    'std': timing_metrics.get('std_duration', 0)
                                }

            if not timing_data:
                print("⚠️ No timing data found in reproducibility results")
                return

            # Create simple mean execution time plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Prepare data
            configs = list(timing_data.keys())
            mean_times = [timing_data[config]['mean'] for config in configs]
            std_times = [timing_data[config]['std'] for config in configs]

            # Define colors by precision type
            colors = []
            for config in configs:
                if 'FP32' in config:
                    colors.append('#1f77b4')  # Blue
                elif 'FP16' in config:
                    colors.append('#ff7f0e')  # Orange
                elif 'BF16' in config:
                    colors.append('#2ca02c')  # Green
                elif 'TF32' in config:
                    colors.append('#d62728')  # Red
                else:
                    colors.append('#9467bd')  # Purple

            # Create bar chart
            bars = ax.bar(range(len(configs)), mean_times, yerr=std_times,
                         capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

            # Formatting
            ax.set_xlabel('Configuration', fontsize=16, fontweight='bold')
            ax.set_ylabel('Mean Execution Time (seconds)', fontsize=16, fontweight='bold')
            ax.set_title('Mean Execution Time by Configuration\n(Cold Start Excluded)',
                        fontsize=18, fontweight='bold', pad=20)

            # Clean up configuration names
            clean_names = []
            for config in configs:
                clean_name = config.replace('_', ' ').replace('Deterministic', 'Det').replace('Non-Deterministic', 'Non-Det')
                clean_names.append(clean_name)

            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(clean_names, rotation=45, ha='right', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for i, (bar, mean_time, std_time) in enumerate(zip(bars, mean_times, std_times)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std_time + max(mean_times) * 0.02,
                       f'{mean_time:.3f}s', ha='center', va='bottom',
                       fontsize=11, fontweight='bold')

            # Make plot look clean and professional
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

            plt.tight_layout()

            # Save the plot
            mean_time_plot_path = self.analyze_dir / 'mean_execution_time_plot.png'
            plt.savefig(mean_time_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"⏱️ Mean execution time plot saved to: {mean_time_plot_path}")

            # Print simple summary
            print("\n📊 Mean Execution Time Summary:")
            print("-" * 40)
            sorted_configs = sorted(timing_data.items(), key=lambda x: x[1]['mean'])
            for i, (config, data) in enumerate(sorted_configs, 1):
                print(f"{i}. {config}: {data['mean']:.3f}s ± {data['std']:.3f}s")

        except Exception as e:
            print(f"❌ Error generating mean execution time plot: {e}")
            import traceback
            traceback.print_exc()


    def _generate_performance_summary(self, timing_data: Dict):
        """Generate a concise performance summary for the standalone plot"""
        try:
            print("\n📊 Performance Summary:")
            print("-" * 50)

            # Sort configurations by performance
            sorted_configs = sorted(timing_data.items(), key=lambda x: x[1]['mean'])

            # Find fastest and slowest
            fastest = sorted_configs[0]
            slowest = sorted_configs[-1]

            print(f"🚀 Fastest: {fastest[0]} - {fastest[1]['mean']:.3f}s ± {fastest[1]['std']:.3f}s")
            print(f"🐌 Slowest: {slowest[0]} - {slowest[1]['mean']:.3f}s ± {slowest[1]['std']:.3f}s")

            if len(sorted_configs) > 1:
                speedup = slowest[1]['mean'] / fastest[1]['mean']
                print(f"⚡ Max Speedup: {speedup:.1f}× ({fastest[0]} vs {slowest[0]})")

            # Calculate coefficient of variation for each config
            print("\n📈 Stability Ranking (Lower CV = More Stable):")
            stability_ranked = sorted(timing_data.items(),
                                    key=lambda x: (x[1]['std'] / x[1]['mean']) if x[1]['mean'] > 0 else float('inf'))

            for i, (config, data) in enumerate(stability_ranked[:3], 1):
                cv = (data['std'] / data['mean'] * 100) if data['mean'] > 0 else 0
                print(f"  {i}. {config}: CV = {cv:.1f}%")

        except Exception as e:
            print(f"❌ Error generating performance summary: {e}")

    def generate_timing_plot(self):
        """Generate timing comparison plots for different configurations"""
        if not PLOTTING_AVAILABLE:
            print("⚠️ Skipping timing plot generation - plotting libraries not available")
            return

        if not self.reproducibility_data:
            print("⚠️ No reproducibility data available for timing plots")
            return

        try:
            print("⏱️ Generating timing comparison plots...")

            # Extract timing data from reproducibility results
            timing_data = {}
            individual_timings = {}

            for config_name, config_results in self.reproducibility_data.items():
                if isinstance(config_results, dict) and 'metrics' in config_results:
                    # Get individual timings if available (exclude first run for cold start)
                    if 'timings' in config_results:
                        raw_timings = config_results['timings']
                        if len(raw_timings) > 1:
                            # Exclude first run to avoid cold start bias
                            warm_timings = raw_timings[1:]
                            individual_timings[config_name] = warm_timings

                            # Recalculate statistics without cold start
                            mean_duration = np.mean(warm_timings)
                            std_duration = np.std(warm_timings)
                            total_duration = np.sum(warm_timings)

                            timing_data[config_name] = {
                                'mean': mean_duration,
                                'std': std_duration,
                                'total': total_duration,
                                'cold_start_time': raw_timings[0],
                                'num_warm_runs': len(warm_timings)
                            }

                            print(f"   📊 {config_name}: Excluded cold start time ({raw_timings[0]:.3f}s), using {len(warm_timings)} warm runs")
                        else:
                            # Not enough data to exclude cold start
                            individual_timings[config_name] = raw_timings
                            timing_metrics = config_results['metrics'].get('timing', {})
                            if timing_metrics:
                                timing_data[config_name] = {
                                    'mean': timing_metrics.get('mean_duration', 0),
                                    'std': timing_metrics.get('std_duration', 0),
                                    'total': timing_metrics.get('total_duration', 0),
                                    'cold_start_time': None,
                                    'num_warm_runs': len(raw_timings)
                                }
                            print(f"   ⚠️ {config_name}: Not enough runs to exclude cold start (only {len(raw_timings)} runs)")
                    else:
                        # Fallback to summary statistics if individual timings not available
                        timing_metrics = config_results['metrics'].get('timing', {})
                        if timing_metrics:
                            timing_data[config_name] = {
                                'mean': timing_metrics.get('mean_duration', 0),
                                'std': timing_metrics.get('std_duration', 0),
                                'total': timing_metrics.get('total_duration', 0),
                                'cold_start_time': None,
                                'num_warm_runs': None
                            }
                            print(f"   ℹ️ {config_name}: Using summary statistics (individual timings not available)")

            if not timing_data:
                print("⚠️ No timing data found in reproducibility results")
                return

            # Create timing comparison plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # 1. Mean execution time comparison
            configs = list(timing_data.keys())
            mean_times = [timing_data[config]['mean'] for config in configs]
            std_times = [timing_data[config]['std'] for config in configs]

            # Color by precision type
            colors = []
            for config in configs:
                if 'FP32' in config:
                    colors.append('#1f77b4')  # Blue
                elif 'FP16' in config:
                    colors.append('#ff7f0e')  # Orange
                elif 'BF16' in config:
                    colors.append('#2ca02c')  # Green
                elif 'TF32' in config:
                    colors.append('#d62728')  # Red
                else:
                    colors.append('#9467bd')  # Purple

            bars1 = ax1.bar(range(len(configs)), mean_times, yerr=std_times,
                           capsize=5, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Mean Execution Time (seconds)', fontsize=12, fontweight='bold')
            ax1.set_title('Mean Execution Time by Configuration\n(Cold Start Excluded)', fontsize=14, fontweight='bold')
            ax1.set_xticks(range(len(configs)))
            ax1.set_xticklabels([config.replace(' ', '\n') for config in configs],
                               rotation=45, ha='right', fontsize=10)
            ax1.grid(True, alpha=0.3)            # Add value labels on bars
            for i, (bar, mean_time, std_time) in enumerate(zip(bars1, mean_times, std_times)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + std_time + 0.01,
                        f'{mean_time:.3f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

            # 2. Total execution time comparison
            total_times = [timing_data[config]['total'] for config in configs]
            bars2 = ax2.bar(range(len(configs)), total_times, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Total Execution Time (seconds)', fontsize=12, fontweight='bold')
            ax2.set_title('Total Execution Time by Configuration\n(Cold Start Excluded)', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(configs)))
            ax2.set_xticklabels([config.replace(' ', '\n') for config in configs],
                               rotation=45, ha='right', fontsize=10)
            ax2.grid(True, alpha=0.3)            # Add value labels on bars
            for i, (bar, total_time) in enumerate(zip(bars2, total_times)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{total_time:.3f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

            # 3. Timing variability (coefficient of variation)
            cv_values = []
            for config in configs:
                mean_val = timing_data[config]['mean']
                std_val = timing_data[config]['std']
                cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                cv_values.append(cv)

            bars3 = ax3.bar(range(len(configs)), cv_values, color=colors, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Configuration', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
            ax3.set_title('Timing Variability by Configuration\n(Cold Start Excluded)', fontsize=14, fontweight='bold')
            ax3.set_xticks(range(len(configs)))
            ax3.set_xticklabels([config.replace(' ', '\n') for config in configs],
                               rotation=45, ha='right', fontsize=10)
            ax3.grid(True, alpha=0.3)            # Add value labels on bars
            for i, (bar, cv) in enumerate(zip(bars3, cv_values)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{cv:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

            # 4. Individual timing distributions (box plot)
            if individual_timings:
                timing_values = []
                timing_labels = []
                for config in configs:
                    if config in individual_timings:
                        timing_values.append(individual_timings[config])
                        timing_labels.append(config.replace(' ', '\n'))

                if timing_values:
                    box_plot = ax4.boxplot(timing_values, labels=timing_labels, patch_artist=True)

                    # Color the boxes
                    for patch, color in zip(box_plot['boxes'], colors[:len(timing_values)]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax4.set_xlabel('Configuration', fontsize=12, fontweight='bold')
                    ax4.set_ylabel('Individual Run Times (seconds)', fontsize=12, fontweight='bold')
                    ax4.set_title('Timing Distribution by Configuration\n(Cold Start Excluded)', fontsize=14, fontweight='bold')
                    ax4.tick_params(axis='x', rotation=45, labelsize=10)
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'No individual timing data available',
                            ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                    ax4.set_title('Individual Timing Distributions', fontsize=14, fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'No individual timing data available',
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Individual Timing Distributions', fontsize=14, fontweight='bold')

            # Overall plot formatting
            plt.suptitle('Embedding Generation Timing Analysis\n(Cold Start Times Excluded)', fontsize=18, fontweight='bold', y=0.98)
            plt.tight_layout()

            # Save timing plot
            timing_output_path = self.analyze_dir / 'embedding_timing_analysis.png'
            plt.savefig(timing_output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"⏱️ Timing analysis plot saved to: {timing_output_path}")

            # Generate timing summary statistics
            self._generate_timing_summary(timing_data, individual_timings)

        except Exception as e:
            print(f"❌ Error generating timing plots: {e}")
            import traceback
            traceback.print_exc()

    def _generate_timing_summary(self, timing_data: Dict, individual_timings: Dict):
        """Generate and save timing summary statistics"""
        try:
            print("📊 Generating timing summary statistics...")

            # Calculate performance metrics
            summary_stats = {}

            for config_name, timings in timing_data.items():
                mean_time = timings['mean']
                std_time = timings['std']
                total_time = timings['total']

                # Calculate texts per second (assuming 10 texts as shown in the reproducibility script)
                num_texts = 10  # This should match the number of texts in the test
                texts_per_second = num_texts / mean_time if mean_time > 0 else 0
                time_per_text = mean_time / num_texts if num_texts > 0 else mean_time

                cv = (std_time / mean_time * 100) if mean_time > 0 else 0

                summary_stats[config_name] = {
                    'mean_time_seconds': mean_time,
                    'std_time_seconds': std_time,
                    'total_time_seconds': total_time,
                    'coefficient_of_variation_percent': cv,
                    'texts_per_second': texts_per_second,
                    'time_per_text_seconds': time_per_text,
                    'cold_start_time_seconds': timings.get('cold_start_time'),
                    'num_warm_runs': timings.get('num_warm_runs')
                }

            # Sort by mean execution time
            sorted_configs = sorted(summary_stats.items(), key=lambda x: x[1]['mean_time_seconds'])

            # Save detailed timing summary
            timing_summary_file = self.analyze_dir / 'timing_summary.json'
            with open(timing_summary_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'note': 'Cold start times (first run) were excluded from statistics to avoid initialization bias',
                    'summary_statistics': dict(sorted_configs),
                    'fastest_configuration': sorted_configs[0][0] if sorted_configs else None,
                    'slowest_configuration': sorted_configs[-1][0] if sorted_configs else None,
                    'individual_timings_warm_only': individual_timings
                }, f, indent=2, cls=NumpyEncoder)

            print(f"📊 Timing summary saved to: {timing_summary_file}")

            # Print performance ranking
            print("\n⚡ Performance Ranking (Fastest to Slowest, Cold Start Excluded):")
            for i, (config_name, stats) in enumerate(sorted_configs, 1):
                mean_time = stats['mean_time_seconds']
                texts_per_sec = stats['texts_per_second']
                cv = stats['coefficient_of_variation_percent']
                cold_start = stats.get('cold_start_time_seconds')
                num_warm = stats.get('num_warm_runs')

                print(f"  {i}. {config_name}")
                print(f"     Mean Time: {mean_time:.3f}s | Texts/sec: {texts_per_sec:.1f} | CV: {cv:.1f}%")
                if cold_start is not None:
                    print(f"     Cold Start: {cold_start:.3f}s | Warm Runs: {num_warm}")
                else:
                    print(f"     Warm Runs: {num_warm if num_warm else 'N/A'}")

        except Exception as e:
            print(f"❌ Error generating timing summary: {e}")

    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting Embedding Uncertainty Analysis")
        print("=" * 60)

        # Load data
        repro_loaded = self.load_embedding_reproducibility_results()
        precision_loaded = self.parse_precision_comparison_report()

        if not repro_loaded and not precision_loaded:
            print("❌ No data sources loaded. Please ensure results files exist.")
            return False

        if not repro_loaded:
            print("⚠️ Proceeding with precision data only")
        if not precision_loaded:
            print("⚠️ Proceeding with reproducibility data only")

        # Generate analysis
        analysis = self.generate_comprehensive_analysis()

        # Save results
        self.save_analysis_results(analysis)

        # Generate heatmaps for precision comparisons
        self.generate_precision_heatmaps(analysis)

        # Generate mean execution time plot (simple focused view)
        self.generate_mean_execution_time_plot()

        # Generate comprehensive timing analysis plots
        self.generate_timing_plot()

        print("\n" + "=" * 60)
        print("✅ Analysis Complete!")

        # Print key findings
        summary = analysis.get('summary_findings', {})
        key_findings = summary.get('key_findings', [])
        if key_findings:
            print("\n🎯 Key Findings:")
            for finding in key_findings[:5]:  # Show top 5 findings
                print(f"  {finding}")

        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print("\n💡 Top Recommendations:")
            for rec in recommendations[:3]:  # Show top 3 recommendations
                print(f"  {rec}")

        print(f"\n📁 Results saved to: {self.analyze_dir}")
        return True


def generate_mean_execution_time_only(results_dir="results"):
    """
    Standalone function to generate only the mean execution time plot

    Args:
        results_dir: Directory containing the embedding_reproducibility_results.json file

    Returns:
        bool: True if successful, False otherwise
    """
    print("⏱️ Generating Mean Execution Time Plot")
    print("=" * 40)

    analyzer = EmbeddingUncertaintyAnalyzer(results_dir)

    # Load only reproducibility data
    if not analyzer.load_embedding_reproducibility_results():
        print("❌ Failed to load embedding reproducibility results.")
        print("   Make sure 'embedding_reproducibility_results.json' exists in the results directory.")
        return False

    # Generate just the mean execution time plot
    analyzer.generate_mean_execution_time_plot()

    print("\n✅ Mean execution time plot generated successfully!")
    print(f"📁 Check: {analyzer.analyze_dir}/mean_execution_time_plot.png")
    return True


def main():
    """Main function to run the analysis"""
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--mean-time-only":
            results_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
            success = generate_mean_execution_time_only(results_dir)
            if not success:
                sys.exit(1)
            return
        elif sys.argv[1] == "--timing-only":
            results_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
            success = generate_standalone_timing_plot_only(results_dir)
            if not success:
                sys.exit(1)
            return

    # Run full analysis
    analyzer = EmbeddingUncertaintyAnalyzer("results")
    success = analyzer.run_analysis()

    if success:
        print("\n🎉 Analysis completed successfully!")
        print("Check the 'results/analyze/' directory for detailed results.")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
