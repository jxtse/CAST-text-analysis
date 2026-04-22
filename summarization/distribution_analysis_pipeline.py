import os
import json
import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sentence_transformers import SentenceTransformer
from collections import Counter, defaultdict
from datetime import datetime

 
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DistributionAnalyzer:
    """Analyzes distribution patterns in LLM outputs."""
    def __init__(self, log_path="Output/Distribution-Analysis/log.txt"):
        directory_path = os.path.dirname(log_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
        self.logger = self._create_simple_logger(log_path)
        
         
        print("Loading sentence embedding model...")
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Sentence embedding model loaded successfully")
        except Exception as e:
            print(f"Failed to load sentence embedding model: {e}")
            self.sentence_model = None
        
         
        self.output_dir = "Output/Distribution-Analysis"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _create_simple_logger(self, log_path):
        """Create a simple logger for distribution analysis."""
        class SimpleLogger:
            def __init__(self, log_path):
                self.log_path = log_path
                directory_path = os.path.dirname(log_path)
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                self.file = open(log_path, 'a', encoding='utf-8')
            
            def write_line(self, message):
                """Write message to both console and log file."""
                print(message)
                self.file.write(message + '\n')
                self.file.flush()
            
            def close(self):
                """Close the log file."""
                self.file.close()
        
        return SimpleLogger(log_path)
    
    def run_distribution_experiments_from_real_data(self):
        """Run distribution analysis using real stability test data."""
        self.logger.write_line(f"[{datetime.now()}] [INFO] Starting distribution analysis from real stability data")
        
         
        stability_files = self._find_stability_result_files()
        
        if not stability_files:
            self.logger.write_line(f"[{datetime.now()}] [ERROR] No stability test result files found")
            print("Error: No stability test result files found. Please run stability pipeline first to generate data.")
            return
        
         
        self._analyze_discrete_distributions_from_real_data(stability_files)
        
         
        if self.sentence_model:
            self._analyze_text_distributions_from_real_data(stability_files)
        
         
        self._analyze_intermediate_states_from_real_data(stability_files)
        
         
        self._generate_comprehensive_report_from_real_data(stability_files)
        
        self.logger.write_line(f"[{datetime.now()}] [INFO] 基于真实数据的分布分析实验完成")
        self.logger.close()
    
    def _find_stability_result_files(self):
        """查找现有的稳定性测试结果文件"""
        stability_files = {}
        
         
        search_patterns = [
            "Output/Stability-Output/{prompt_type}_stability_results.json",
            "Output/Stability-Output/reasoning_path_comparison/{prompt_type}_stability_results.json",
            "Output/Stability-Output/extended_cast_comparison/{prompt_type}_stability_results.json"
        ]
        
         
        original_prompt_types = ["perspective_prompt", "num_of_bullet_points_prompt", "domain_prompt", "num_of_text_items_prompt"]
        
         
        extended_prompt_types = ["full_cast_prompt", "minimal_prompt", "topwords_only_prompt"]
        
         
        all_prompt_types = original_prompt_types + extended_prompt_types
        
        for prompt_type in all_prompt_types:
            found = False
            for pattern in search_patterns:
                file_path = pattern.format(prompt_type=prompt_type)
                if os.path.exists(file_path):
                    stability_files[prompt_type] = file_path
                    self.logger.write_line(f"[{datetime.now()}] [INFO] 找到文件: {file_path}")
                    found = True
                    break
            
            if not found:
                self.logger.write_line(f"[{datetime.now()}] [WARNING] 未找到 {prompt_type} 的稳定性结果文件")
        
        return stability_files
    
    def _analyze_discrete_distributions_from_real_data(self, stability_files):
        """从真实数据分析离散型分布"""
        self.logger.write_line(f"[{datetime.now()}] [INFO] 开始分析真实数据的离散型分布")
        
         
        discrete_features = {
            'num_bullet_points': self._extract_bullet_point_count_from_real,
            'text_length': self._extract_text_length_from_real,
            'num_words': self._extract_word_count_from_real
        }
        
        discrete_results = {}
        
        for feature_name, extractor_func in discrete_features.items():
            self.logger.write_line(f"[{datetime.now()}] [INFO] 分析特征: {feature_name}")
            discrete_results[feature_name] = {}
            
            for prompt_type, file_path in stability_files.items():
                self.logger.write_line(f"[{datetime.now()}] [INFO] 处理prompt类型: {prompt_type}")
                
                 
                datasets = self._load_stability_results(file_path)
                
                 
                discrete_results[feature_name][prompt_type] = {}
                
                for dataset_name, real_data in datasets.items():
                    feature_values = []
                    for sample in real_data:
                        try:
                            value = extractor_func(sample)
                            if value is not None:
                                feature_values.append(value)
                        except Exception as e:
                            self.logger.write_line(f"[{datetime.now()}] [WARNING] 提取特征失败: {str(e)}")
                            continue
                    
                    discrete_results[feature_name][prompt_type][dataset_name] = feature_values
                    
                    self.logger.write_line(f"[{datetime.now()}] [INFO] {prompt_type} - {dataset_name} - {feature_name}: 获得 {len(feature_values)} 个有效样本")
        
         
        with open(f"{self.output_dir}/real_discrete_raw_data.json", "w", encoding="utf-8") as f:
            json.dump(discrete_results, f, indent=2, ensure_ascii=False)
        
         
        self._visualize_discrete_distributions(discrete_results, prefix="real_")
        self._calculate_discrete_distribution_metrics(discrete_results, prefix="real_")
    
    def _load_stability_results(self, file_path):
        """加载稳定性测试结果，保留数据集分组信息"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
             
            datasets = {}
            if isinstance(data, list):
                for group_idx, group in enumerate(data):
                    if isinstance(group, list):
                        datasets[f"dataset_{group_idx}"] = group
                    elif isinstance(group, dict):
                        datasets[f"dataset_{group_idx}"] = [group]
            else:
                datasets["dataset_0"] = [data] if isinstance(data, dict) else []
            
            return datasets
        except Exception as e:
            self.logger.write_line(f"[{datetime.now()}] [ERROR] 加载文件失败 {file_path}: {str(e)}")
            return {}
    
    def _extract_bullet_point_count_from_real(self, sample):
        """从真实数据提取bullet point数量"""
        bullet_list = sample.get('BulletPoint', [])
        return len(bullet_list) if bullet_list else 0
    
    def _extract_text_length_from_real(self, sample):
        """从真实数据提取文本长度"""
        analysis_result = sample.get('AnalysisResult', '')
        return len(analysis_result) if analysis_result else 0
    
    def _extract_word_count_from_real(self, sample):
        """从真实数据提取单词/字符数量"""
        analysis_result = sample.get('AnalysisResult', '')
        if not analysis_result:
            return 0
        
         
        import string
         
        words = analysis_result.translate(str.maketrans('', '', string.punctuation)).split()
        return len(words)
    
    def _analyze_text_distributions_from_real_data(self, stability_files):
        """从真实数据分析文本型分布"""
        self.logger.write_line(f"[{datetime.now()}] [INFO] 开始分析真实数据的文本型分布")
        
        text_results = {}
        
        for prompt_type, file_path in stability_files.items():
            self.logger.write_line(f"[{datetime.now()}] [INFO] 处理prompt类型: {prompt_type}")
            
             
            datasets = self._load_stability_results(file_path)
            
             
            text_results[prompt_type] = {}
            
            for dataset_name, real_data in datasets.items():
                texts = []
                for sample in real_data:
                    analysis_result = sample.get('AnalysisResult', '')
                    if analysis_result:
                        texts.append(analysis_result)
                
                text_results[prompt_type][dataset_name] = texts
                self.logger.write_line(f"[{datetime.now()}] [INFO] {prompt_type} - {dataset_name}: 获得 {len(texts)} 个文本样本")
        
         
        has_enough_data = False
        for prompt_type, dataset_data in text_results.items():
            if isinstance(dataset_data, dict):
                for dataset_name, texts in dataset_data.items():
                    if len(texts) > 5:
                        has_enough_data = True
                        break
            if has_enough_data:
                break
        
        if has_enough_data:
            try:
                self._visualize_text_distributions_real(text_results)
            except Exception as e:
                self.logger.write_line(f"[{datetime.now()}] [ERROR] 文本分布可视化失败: {str(e)}")
    
    def _visualize_text_distributions_real(self, text_results):
        """可视化真实文本数据的分布，所有数据集在一张图上用不同颜色标识"""
        self.logger.write_line(f"[{datetime.now()}] [INFO] 开始文本分布可视化")
        
         
        all_datasets = set()
        all_prompt_types = set()
        for prompt_type, dataset_data in text_results.items():
            all_prompt_types.add(prompt_type)
            if isinstance(dataset_data, dict):
                all_datasets.update(dataset_data.keys())
        
        all_datasets = sorted(list(all_datasets))
        all_prompt_types = sorted(list(all_prompt_types))
        
        if not all_datasets or not all_prompt_types:
            self.logger.write_line(f"[{datetime.now()}] [WARNING] 没有足够的数据集或prompt类型进行文本分布可视化")
            return
        
         
        for dataset_name in all_datasets:
            self.logger.write_line(f"[{datetime.now()}] [INFO] 生成 {dataset_name} 的文本分布可视化")
            
             
            all_texts = []
            labels = []
            prompt_colors = {}
            
            for prompt_idx, prompt_type in enumerate(all_prompt_types):
                if prompt_type not in text_results:
                    continue
                    
                dataset_data = text_results[prompt_type]
                if not isinstance(dataset_data, dict) or dataset_name not in dataset_data:
                    continue
                    
                texts = dataset_data[dataset_name]
                if not texts:
                    continue
                    
                 
                sample_texts = texts[:50] if len(texts) > 50 else texts
                all_texts.extend(sample_texts)
                labels.extend([prompt_type] * len(sample_texts))
                
                 
                prompt_colors[prompt_type] = plt.cm.Set3(prompt_idx / len(all_prompt_types))
            
            if len(all_texts) < 10:
                self.logger.write_line(f"[{datetime.now()}] [WARNING] {dataset_name} 文本样本太少，跳过可视化")
                continue
            
             
            self.logger.write_line(f"[{datetime.now()}] [INFO] 生成 {dataset_name} 的 {len(all_texts)} 个文本嵌入向量")
            embeddings = self.sentence_model.encode(all_texts)
            
             
            self.logger.write_line(f"[{datetime.now()}] [INFO] 执行 {dataset_name} 的t-SNE降维")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_texts)-1))
            embeddings_2d = tsne.fit_transform(embeddings)
            
             
            plt.figure(figsize=(14, 10))
            
             
            for prompt_type in all_prompt_types:
                if prompt_type not in prompt_colors:
                    continue
                    
                mask = [label == prompt_type for label in labels]
                if not any(mask):
                    continue
                    
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[prompt_colors[prompt_type]], label=prompt_type, 
                           alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            plt.title(f'Text Distribution Analysis - {dataset_name}\nt-SNE Visualization of Text Embeddings', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/real_text_tsne_distribution_{dataset_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
         
        self._create_combined_text_visualization(text_results, all_datasets, all_prompt_types)
        
        self.logger.write_line(f"[{datetime.now()}] [INFO] 文本分布可视化完成")
    
    def _create_combined_text_visualization(self, text_results, all_datasets, all_prompt_types):
        """创建所有数据集的综合文本分布可视化"""
        self.logger.write_line(f"[{datetime.now()}] [INFO] 创建综合文本分布可视化")
        
         
        all_texts = []
        labels = []
        dataset_labels = []
        
        for prompt_type in all_prompt_types:
            if prompt_type not in text_results:
                continue
                
            dataset_data = text_results[prompt_type]
            if not isinstance(dataset_data, dict):
                continue
                
            for dataset_name in all_datasets:
                if dataset_name not in dataset_data:
                    continue
                    
                texts = dataset_data[dataset_name]
                if not texts:
                    continue
                    
                 
                sample_texts = texts[:30] if len(texts) > 30 else texts
                all_texts.extend(sample_texts)
                labels.extend([prompt_type] * len(sample_texts))
                dataset_labels.extend([dataset_name] * len(sample_texts))
        
        if len(all_texts) < 20:
            self.logger.write_line(f"[{datetime.now()}] [WARNING] 综合文本样本太少，跳过综合可视化")
            return
        
         
        self.logger.write_line(f"[{datetime.now()}] [INFO] 生成综合视图的 {len(all_texts)} 个文本嵌入向量")
        embeddings = self.sentence_model.encode(all_texts)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_texts)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
         
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
         
        prompt_colors = plt.cm.Set3(np.linspace(0, 1, len(all_prompt_types)))
        for i, prompt_type in enumerate(all_prompt_types):
            mask = [label == prompt_type for label in labels]
            if any(mask):
                ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[prompt_colors[i]], label=prompt_type, 
                           alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        ax1.set_title('Text Distribution by Prompt Type\n(All Datasets Combined)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        
         
        dataset_colors = plt.cm.viridis(np.linspace(0, 1, len(all_datasets)))
        dataset_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        for i, dataset_name in enumerate(all_datasets):
            mask = [ds_label == dataset_name for ds_label in dataset_labels]
            if any(mask):
                marker = dataset_markers[i % len(dataset_markers)]
                ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[dataset_colors[i]], label=dataset_name, 
                           marker=marker, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        ax2.set_title('Text Distribution by Dataset\n(All Prompt Types Combined)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        ax2.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Text Distribution Analysis\nt-SNE Visualization of Text Embeddings', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/real_text_tsne_distribution_comprehensive.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_discrete_distributions(self, discrete_results, prefix=""):
        """绘制离散分布可视化图，所有数据集在一张图上用不同颜色标识"""
        for feature_name, prompt_data in discrete_results.items():
            if not prompt_data:
                continue
                
            self.logger.write_line(f"[{datetime.now()}] [INFO] 生成 {feature_name} 的分布对比图")
            
             
            all_datasets = set()
            all_prompt_types = set()
            for prompt_type, dataset_data in prompt_data.items():
                if prompt_type != 'topwords_only_prompt':   
                    all_prompt_types.add(prompt_type)
                    if isinstance(dataset_data, dict):
                        all_datasets.update(dataset_data.keys())
            
            all_datasets = sorted(list(all_datasets))
            all_prompt_types = sorted(list(all_prompt_types))
            
            if not all_datasets or not all_prompt_types:
                continue
            
             
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
             
            plt.style.use('seaborn-v0_8-whitegrid')
            
             
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
            dataset_colors = {dataset: colors[i % len(colors)] for i, dataset in enumerate(all_datasets)}
            
             
            all_values = []
            for prompt_type, dataset_data in prompt_data.items():
                if prompt_type != 'topwords_only_prompt' and isinstance(dataset_data, dict):
                    for dataset_name, values in dataset_data.items():
                        all_values.extend(values)
            
            if not all_values:
                continue
                
            unique_values = sorted(set(all_values))
            
             
            for prompt_idx, prompt_type in enumerate(all_prompt_types):
                if prompt_idx >= 6:   
                    break
                    
                if prompt_type not in prompt_data:
                    continue
                    
                dataset_data = prompt_data[prompt_type]
                if not isinstance(dataset_data, dict):
                    continue
                
                ax = axes[prompt_idx]
                
                 
                ax.set_facecolor('#F8F9FA')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#DDD')
                ax.spines['bottom'].set_color('#DDD')
                
                 
                display_name = prompt_type.replace('_', ' ').title()
                ax.set_title(f'{display_name}\nProbability Distribution', 
                           fontweight='bold', fontsize=12, pad=20,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
                
                 
                bar_width = 0.8 / len(all_datasets)
                
                 
                avg_entropy = 0
                avg_gini = 0
                valid_datasets = 0
                
                for dataset_idx, dataset_name in enumerate(all_datasets):
                    if dataset_name not in dataset_data:
                        continue
                        
                    values = dataset_data[dataset_name]
                    if not values:
                        continue
                    
                    color = dataset_colors[dataset_name]
                    
                    value_counts = Counter(values)
                    probabilities = [value_counts.get(v, 0) / len(values) for v in unique_values]
                    
                     
                    x_pos = [i + dataset_idx * bar_width for i in range(len(unique_values))]
                    
                     
                    bars = ax.bar(x_pos, probabilities, bar_width, 
                                 label=f'{dataset_name} (n={len(values)})', 
                                 color=color, alpha=0.8, edgecolor='white', linewidth=1)
                    
                     
                    if probabilities:
                        entropy_val = entropy([p for p in probabilities if p > 0])
                        gini_val = self._gini_coefficient(np.array(probabilities))
                        avg_entropy += entropy_val
                        avg_gini += gini_val
                        valid_datasets += 1
                
                 
                ax.set_xlabel('Value', fontsize=11, fontweight='bold')
                ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
                ax.set_xticks([i + bar_width * (len(all_datasets) - 1) / 2 for i in range(len(unique_values))])
                ax.set_xticklabels(unique_values)
                ax.tick_params(axis='both', which='major', labelsize=10, colors='#333')
                ax.grid(True, alpha=0.3, linestyle='--', axis='y')
                
                 
                all_probs = []
                for values in dataset_data.values():
                    if values:
                        value_counts = Counter(values)
                        probabilities = [value_counts.get(v, 0) / len(values) for v in unique_values]
                        all_probs.extend(probabilities)
                
                if all_probs:
                    max_prob = max(all_probs)
                    ax.set_ylim(0, max_prob * 1.1)
                else:
                    ax.set_ylim(0, 1.0)
                
                 
                if valid_datasets > 0:
                    avg_entropy /= valid_datasets
                    avg_gini /= valid_datasets
                    stats_text = f'Avg H: {avg_entropy:.3f}\nAvg G: {avg_gini:.3f}'
                    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                           fontsize=10, fontweight='bold', family='monospace')
                
                 
                if prompt_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                             frameon=True, fancybox=True, shadow=True)
            
             
            for i in range(len(all_prompt_types), 6):
                axes[i].set_visible(False)
            
             
            feature_display_name = feature_name.replace('_', ' ').title()
            plt.suptitle(f'{feature_display_name} Distribution Analysis\n'
                        f'Lower Entropy (H) & Higher Gini (G) = Sharper Distribution', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            try:
                plt.tight_layout()
                plt.subplots_adjust(top=0.85)
            except:
                 
                plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9)
            plt.savefig(f"{self.output_dir}/{prefix}distribution_comparison_{feature_name}_all_datasets.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
             
            self._plot_distribution_summary_statistics(prompt_data, feature_name, prefix)
    
    def _plot_distribution_summary_statistics(self, prompt_data, feature_name, prefix=""):
        """绘制分布统计指标的汇总图"""
         
        stats_data = []
        
        for prompt_type, dataset_data in prompt_data.items():
            if prompt_type == 'topwords_only_prompt':   
                continue
                
            if not isinstance(dataset_data, dict):
                continue
                
            for dataset_name, values in dataset_data.items():
                if not values:
                    continue
                    
                value_counts = Counter(values)
                total = len(values)
                probabilities = np.array([count / total for count in value_counts.values()])
                
                stats = {
                    'prompt_type': prompt_type,
                    'dataset': dataset_name,
                    'entropy': entropy(probabilities),
                    'gini_coefficient': self._gini_coefficient(probabilities),
                    'max_probability': float(np.max(probabilities)),
                    'std': float(np.std(values)),
                    'sample_size': len(values)
                }
                stats_data.append(stats)
        
        if not stats_data:
            return
            
         
        df = pd.DataFrame(stats_data)
        
         
        plt.style.use('seaborn-v0_8-whitegrid')
        
         
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        metrics = ['entropy', 'gini_coefficient', 'max_probability', 'std']
        metric_titles = [
            'Entropy (Lower = Sharper)',
            'Gini Coefficient (Higher = More Uneven)',
            'Max Probability (Higher = More Concentrated)',
            'Standard Deviation (Lower = More Concentrated)'
        ]
        
         
        datasets = sorted(df['dataset'].unique())
        prompt_types = sorted(df['prompt_type'].unique())
        
         
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
        color_map = {dataset: colors[i % len(colors)] for i, dataset in enumerate(datasets)}
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx]
            
             
            ax.set_facecolor('#F8F9FA')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#DDD')
            ax.spines['bottom'].set_color('#DDD')
            
             
            x_pos = np.arange(len(prompt_types))
            bar_width = 0.8 / len(datasets)
            
            for dataset_idx, dataset in enumerate(datasets):
                dataset_data = df[df['dataset'] == dataset]
                values = []
                
                for prompt_type in prompt_types:
                    prompt_data = dataset_data[dataset_data['prompt_type'] == prompt_type]
                    if not prompt_data.empty:
                        values.append(prompt_data[metric].iloc[0])
                    else:
                        values.append(0)
                
                bars = ax.bar(x_pos + dataset_idx * bar_width, values,
                             bar_width, label=dataset, color=color_map[dataset], 
                             alpha=0.8, edgecolor='white', linewidth=1.5)
                
                 
                for bar, value in zip(bars, values):
                    if value > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9, 
                               fontweight='bold', color='#333')
            
            ax.set_title(title, fontweight='bold', fontsize=13, pad=15,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.3))
            ax.set_xlabel('Prompt Type', fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos + bar_width * (len(datasets) - 1) / 2)
            
             
            display_labels = [pt.replace('_', ' ').title() for pt in prompt_types]
            ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10, colors='#333')
            
             
            if idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10,
                         frameon=True, fancybox=True, shadow=True)
            
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
             
            if values:
                positive_values = [v for v in values if v > 0]
                if positive_values:
                    max_val = max(positive_values)
                    ax.set_ylim(0, max_val * 1.15)
                else:
                     
                    ax.set_ylim(0, 1.0)
        
         
        feature_display_name = feature_name.replace('_', ' ').replace('intermediate ', '').title()
        plt.suptitle(f'{feature_display_name} Distribution Metrics Summary\n'
                    f'Statistical Comparison Across All Datasets', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        try:
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
        except:
             
            plt.subplots_adjust(top=0.88, bottom=0.15, left=0.1, right=0.9)
        plt.savefig(f"{self.output_dir}/{prefix}distribution_metrics_summary_{feature_name}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _calculate_discrete_distribution_metrics(self, discrete_results, prefix=""):
        """计算离散分布指标，按数据集分离"""
        all_metrics = {}
        
        for feature_name, prompt_data in discrete_results.items():
            if not prompt_data:
                continue
            
             
            all_datasets = set()
            for prompt_type, dataset_data in prompt_data.items():
                if isinstance(dataset_data, dict):
                    all_datasets.update(dataset_data.keys())
            
            all_datasets = sorted(list(all_datasets))
            all_metrics[feature_name] = {}
            
             
            for dataset_name in all_datasets:
                feature_metrics = {}
                
                for prompt_type, dataset_data in prompt_data.items():
                    if not isinstance(dataset_data, dict) or dataset_name not in dataset_data:
                        continue
                        
                    values = dataset_data[dataset_name]
                    if not values:
                        continue
                        
                     
                    value_counts = Counter(values)
                    total = len(values)
                    probabilities = np.array([count / total for count in value_counts.values()])
                    
                     
                    metrics = {
                        'entropy': entropy(probabilities),   
                        'gini_coefficient': self._gini_coefficient(probabilities),   
                        'max_probability': float(np.max(probabilities)),   
                        'variance': float(np.var(values)),   
                        'std': float(np.std(values)),   
                        'mean': float(np.mean(values)),   
                        'unique_values': len(value_counts),   
                        'sample_size': len(values)
                    }
                    
                    feature_metrics[prompt_type] = metrics
                
                all_metrics[feature_name][dataset_name] = feature_metrics
                
                 
                self._plot_metrics_comparison(feature_metrics, f"{feature_name}_{dataset_name}", prefix)
                
                 
                with open(f"{self.output_dir}/{prefix}metrics_{feature_name}_{dataset_name}.json", "w", encoding="utf-8") as f:
                    json.dump(feature_metrics, f, indent=2, ensure_ascii=False)
        
        return all_metrics
    
    def _plot_metrics_comparison(self, metrics, feature_name, prefix=""):
        """绘制指标对比图"""
        if not metrics:
            return
            
        self.logger.write_line(f"[{datetime.now()}] [INFO] 生成 {feature_name} 的指标对比图")
        
         
        prompt_types = list(metrics.keys())
        
         
        key_metrics = {
            'entropy': 'Entropy (lower = sharper)',
            'gini_coefficient': 'Gini Coefficient (higher = more uneven)',
            'max_probability': 'Max Probability (higher = more concentrated)',
            'std': 'Standard Deviation (lower = more concentrated)'
        }
        
         
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for idx, (metric_key, metric_name) in enumerate(key_metrics.items()):
            values = [metrics[pt][metric_key] for pt in prompt_types if metric_key in metrics[pt]]
            valid_types = [pt for pt in prompt_types if metric_key in metrics[pt]]
            
            if not values:
                continue
                
            bars = axes[idx].bar(valid_types, values, 
                               color=colors[:len(valid_types)], alpha=0.7)
            axes[idx].set_title(metric_name, fontweight='bold')
            axes[idx].set_ylabel('Value')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
            
             
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                             f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(f'{feature_name} Metrics Comparison (Real Data)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{prefix}metrics_comparison_{feature_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _gini_coefficient(self, probabilities):
        """计算基尼系数"""
        if len(probabilities) == 0:
            return 0
        sorted_probs = np.sort(probabilities)
        n = len(sorted_probs)
        cumsum = np.cumsum(sorted_probs)
        return (n + 1 - 2 * np.sum(cumsum)) / (n * np.sum(sorted_probs))
    
    def _analyze_intermediate_states_from_real_data(self, stability_files):
        """分析真实数据中的中间状态分布"""
        self.logger.write_line(f"[{datetime.now()}] [INFO] 开始分析中间状态分布")
        
         
        intermediate_features = {
            'domain_consistency': self._extract_domain_consistency,
            'topwords_diversity': self._extract_topwords_diversity,
            'perspective_coherence': self._extract_perspective_coherence,
            'reasoning_structure': self._extract_reasoning_structure
        }
        
        intermediate_results = {}
        
        for feature_name, extractor_func in intermediate_features.items():
            self.logger.write_line(f"[{datetime.now()}] [INFO] 分析中间状态特征: {feature_name}")
            intermediate_results[feature_name] = {}
            
            for prompt_type, file_path in stability_files.items():
                self.logger.write_line(f"[{datetime.now()}] [INFO] 处理prompt类型: {prompt_type}")
                
                 
                datasets = self._load_stability_results(file_path)
                
                 
                intermediate_results[feature_name][prompt_type] = {}
                
                for dataset_name, real_data in datasets.items():
                    feature_values = []
                    for sample in real_data:
                        try:
                            value = extractor_func(sample)
                            if value is not None:
                                feature_values.append(value)
                        except Exception as e:
                            self.logger.write_line(f"[{datetime.now()}] [WARNING] 提取中间状态特征失败: {str(e)}")
                            continue
                    
                    intermediate_results[feature_name][prompt_type][dataset_name] = feature_values
                    
                    self.logger.write_line(f"[{datetime.now()}] [INFO] {prompt_type} - {dataset_name} - {feature_name}: 获得 {len(feature_values)} 个有效样本")
        
         
        with open(f"{self.output_dir}/intermediate_states_raw_data.json", "w", encoding="utf-8") as f:
            json.dump(intermediate_results, f, indent=2, ensure_ascii=False)
        
         
        self._visualize_intermediate_distributions(intermediate_results)
        self._calculate_intermediate_distribution_metrics(intermediate_results)
        
        return intermediate_results
    
    def _extract_domain_consistency(self, sample):
        """提取Domain字段的一致性特征"""
        try:
             
            analysis_result = sample.get('AnalysisResult', '')
            if '"Domain"' in analysis_result or '"domain"' in analysis_result.lower():
                 
                import re
                domain_match = re.search(r'"[Dd]omain[^"]*":\s*"([^"]+)"', analysis_result)
                if domain_match:
                    domain_text = domain_match.group(1)
                    return len(domain_text.split())   
            return 0
        except Exception:
            return 0
    
    def _extract_topwords_diversity(self, sample):
        """提取TopWords的多样性特征"""
        try:
            analysis_result = sample.get('AnalysisResult', '')
            if '"TopWords"' in analysis_result or '"topwords"' in analysis_result.lower():
                import re
                 
                topwords_match = re.search(r'"[Tt]op[Ww]ords[^"]*":\s*\[([^\]]+)\]', analysis_result)
                if topwords_match:
                    topwords_text = topwords_match.group(1)
                     
                    keywords = [word.strip().strip('"') for word in topwords_text.split(',')]
                    return len([w for w in keywords if w])   
            return 0
        except Exception:
            return 0
    
    def _extract_perspective_coherence(self, sample):
        """提取Perspective的连贯性特征"""
        try:
            analysis_result = sample.get('AnalysisResult', '')
            if '"Perspective"' in analysis_result or '"NumTopics"' in analysis_result:
                import re
                 
                numtopics_match = re.search(r'"[Nn]um[Tt]opics[^"]*":\s*"?(\d+)"?', analysis_result)
                if numtopics_match:
                    return int(numtopics_match.group(1))
            return 0
        except Exception:
            return 0
    
    def _extract_reasoning_structure(self, sample):
        """提取推理结构复杂度特征"""
        try:
            analysis_result = sample.get('AnalysisResult', '')
             
            structured_fields = ['Domain', 'Perspective', 'TopWords', 'AnalysisSteps', 'ProcessingNotes']
            structure_count = sum(1 for field in structured_fields if field in analysis_result)
            return structure_count
        except Exception:
            return 0
    
    def _visualize_intermediate_distributions(self, intermediate_results):
        """可视化中间状态分布，所有数据集在一张图上用不同颜色标识"""
        for feature_name, prompt_data in intermediate_results.items():
            if not prompt_data:
                continue
            
            self.logger.write_line(f"[{datetime.now()}] [INFO] 生成 {feature_name} 的中间状态分布图")
            
             
            all_datasets = set()
            all_prompt_types = set()
            for prompt_type, dataset_data in prompt_data.items():
                if prompt_type != 'topwords_only_prompt':   
                    all_prompt_types.add(prompt_type)
                    if isinstance(dataset_data, dict):
                        all_datasets.update(dataset_data.keys())
            
            all_datasets = sorted(list(all_datasets))
            all_prompt_types = sorted(list(all_prompt_types))
            
            if not all_datasets or not all_prompt_types:
                continue
            
             
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
             
            plt.style.use('seaborn-v0_8-whitegrid')
            
             
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
            dataset_colors = {dataset: colors[i % len(colors)] for i, dataset in enumerate(all_datasets)}
            
             
            all_values = []
            for prompt_type, dataset_data in prompt_data.items():
                if prompt_type != 'topwords_only_prompt' and isinstance(dataset_data, dict):
                    for dataset_name, values in dataset_data.items():
                        all_values.extend(values)
            
            if not all_values:
                continue
                
            unique_values = sorted(set(all_values))
            
             
            for prompt_idx, prompt_type in enumerate(all_prompt_types):
                if prompt_idx >= 6:   
                    break
                    
                if prompt_type not in prompt_data:
                    continue
                    
                dataset_data = prompt_data[prompt_type]
                if not isinstance(dataset_data, dict):
                    continue
                
                ax = axes[prompt_idx]
                
                 
                ax.set_facecolor('#F8F9FA')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#DDD')
                ax.spines['bottom'].set_color('#DDD')
                
                 
                display_name = prompt_type.replace('_', ' ').title()
                ax.set_title(f'{display_name}\n{feature_name.replace("_", " ").title()} Distribution', 
                           fontweight='bold', fontsize=12, pad=20,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.5))
                
                 
                bar_width = 0.8 / len(all_datasets)
                
                 
                avg_entropy = 0
                avg_gini = 0
                valid_datasets = 0
                
                for dataset_idx, dataset_name in enumerate(all_datasets):
                    if dataset_name not in dataset_data:
                        continue
                        
                    values = dataset_data[dataset_name]
                    if not values:
                        continue
                    
                    color = dataset_colors[dataset_name]
                    
                    value_counts = Counter(values)
                    probabilities = [value_counts.get(v, 0) / len(values) for v in unique_values]
                    
                     
                    x_pos = [i + dataset_idx * bar_width for i in range(len(unique_values))]
                    
                     
                    bars = ax.bar(x_pos, probabilities, bar_width, 
                                 label=f'{dataset_name} (n={len(values)})', 
                                 color=color, alpha=0.8, edgecolor='white', linewidth=1)
                    
                     
                    if probabilities:
                        entropy_val = entropy([p for p in probabilities if p > 0])
                        gini_val = self._gini_coefficient(np.array(probabilities))
                        avg_entropy += entropy_val
                        avg_gini += gini_val
                        valid_datasets += 1
                
                 
                ax.set_xlabel('Value', fontsize=11, fontweight='bold')
                ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
                ax.set_xticks([i + bar_width * (len(all_datasets) - 1) / 2 for i in range(len(unique_values))])
                ax.set_xticklabels(unique_values)
                ax.tick_params(axis='both', which='major', labelsize=10, colors='#333')
                ax.grid(True, alpha=0.3, linestyle='--', axis='y')
                
                 
                max_prob = 0
                for dataset_name, values in dataset_data.items():
                    if values:
                        value_counts = Counter(values)
                        probabilities = [value_counts.get(v, 0) / len(values) for v in unique_values]
                        if probabilities:
                            max_prob = max(max_prob, max(probabilities))
                
                if max_prob > 0:
                    ax.set_ylim(0, max_prob * 1.1)
                else:
                    ax.set_ylim(0, 1.0)
                
                 
                if valid_datasets > 0:
                    avg_entropy /= valid_datasets
                    avg_gini /= valid_datasets
                    stats_text = f'Avg H: {avg_entropy:.3f}\nAvg G: {avg_gini:.3f}'
                    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                           fontsize=10, fontweight='bold', family='monospace')
                
                 
                if prompt_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                             frameon=True, fancybox=True, shadow=True)
            
             
            for i in range(len(all_prompt_types), 6):
                axes[i].set_visible(False)
            
             
            feature_display_name = feature_name.replace('_', ' ').title()
            plt.suptitle(f'{feature_display_name} Intermediate State Analysis\n'
                        f'Lower Entropy (H) & Higher Gini (G) = Sharper Distribution', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            try:
                plt.tight_layout()
                plt.subplots_adjust(top=0.85)
            except:
                 
                plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9)
            plt.savefig(f"{self.output_dir}/intermediate_distribution_{feature_name}_all_datasets.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
             
            self._plot_distribution_summary_statistics(prompt_data, f"intermediate_{feature_name}", "intermediate_")
    
    def _calculate_intermediate_distribution_metrics(self, intermediate_results):
        """计算中间状态分布指标，按数据集分离"""
        all_intermediate_metrics = {}
        
        for feature_name, prompt_data in intermediate_results.items():
            if not prompt_data:
                continue
            
             
            all_datasets = set()
            for prompt_type, dataset_data in prompt_data.items():
                if isinstance(dataset_data, dict):
                    all_datasets.update(dataset_data.keys())
            
            all_datasets = sorted(list(all_datasets))
            all_intermediate_metrics[feature_name] = {}
            
             
            for dataset_name in all_datasets:
                feature_metrics = {}
                
                for prompt_type, dataset_data in prompt_data.items():
                    if not isinstance(dataset_data, dict) or dataset_name not in dataset_data:
                        continue
                        
                    values = dataset_data[dataset_name]
                    if not values:
                        continue
                    
                     
                    value_counts = Counter(values)
                    total = len(values)
                    probabilities = np.array([count / total for count in value_counts.values()])
                    
                     
                    metrics = {
                        'entropy': entropy(probabilities),   
                        'gini_coefficient': self._gini_coefficient(probabilities),   
                        'max_probability': float(np.max(probabilities)),   
                        'variance': float(np.var(values)),   
                        'std': float(np.std(values)),   
                        'mean': float(np.mean(values)),   
                        'unique_values': len(value_counts),   
                        'sample_size': len(values)
                    }
                    
                    feature_metrics[prompt_type] = metrics
                
                all_intermediate_metrics[feature_name][dataset_name] = feature_metrics
                
                 
                with open(f"{self.output_dir}/intermediate_metrics_{feature_name}_{dataset_name}.json", "w", encoding="utf-8") as f:
                    json.dump(feature_metrics, f, indent=2, ensure_ascii=False)
        
        return all_intermediate_metrics
    
    def _generate_comprehensive_report_from_real_data(self, stability_files):
        """生成基于真实数据的综合分析报告"""
        self.logger.write_line(f"[{datetime.now()}] [INFO] 生成综合分析报告")
        
         
        all_metrics = {}
        for feature_name in ['num_bullet_points', 'text_length', 'num_words']:
            all_metrics[feature_name] = {}
             
            import glob
            pattern = f"{self.output_dir}/real_metrics_{feature_name}_*.json"
            metrics_files = glob.glob(pattern)
            
            for metrics_file in metrics_files:
                 
                basename = os.path.basename(metrics_file)
                dataset_name = basename.replace(f"real_metrics_{feature_name}_", "").replace(".json", "")
                
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        all_metrics[feature_name][dataset_name] = json.load(f)
        
         
        intermediate_results = self._analyze_intermediate_states_from_real_data(stability_files)
        
         
        self.logger.write_line(f"[{datetime.now()}] [DEBUG] all_metrics 结构: {list(all_metrics.keys())}")
        self.logger.write_line(f"[{datetime.now()}] [DEBUG] intermediate_results 结构: {list(intermediate_results.keys())}")
        
         
        report = {
            "analysis_summary": {
                "objective": "基于真实稳定性测试数据验证CAST方法的分布尖锐化效果",
                "method": "通过分析真实LLM输出的概率质量函数的信息熵、基尼系数等指标量化分布的尖锐程度",
                "data_source": f"使用了 {len(stability_files)} 种prompt类型的真实稳定性测试数据",
                "interpretation": {
                    "entropy": "信息熵越低表示分布越集中（越尖锐）",
                    "gini_coefficient": "基尼系数越高表示分布越不均匀（越尖锐）",
                    "max_probability": "最大概率越高表示主导值越明显",
                    "std": "标准差越低表示分布越集中"
                }
            },
            "results": {**all_metrics, **intermediate_results},
            "key_findings": [],
            "recommendations": [
                "基于真实数据的分析显示了不同prompt类型的实际输出分布差异",
                "建议结合定性分析验证分布集中是否对应更好的输出质量",
                "可以通过调整prompt的结构化程度来控制输出分布的尖锐程度",
                "需要平衡输出的一致性和多样性，避免过度集中导致输出单调"
            ]
        }
        
         
        for feature_name, dataset_metrics in report["results"].items():
            if not dataset_metrics or not isinstance(dataset_metrics, dict):
                continue
            
             
            for dataset_name, feature_metrics in dataset_metrics.items():
                if not feature_metrics or not isinstance(feature_metrics, dict):
                    continue
                
                 
                valid_metrics = {}
                for pt, metrics in feature_metrics.items():
                    if isinstance(metrics, dict) and 'entropy' in metrics and 'gini_coefficient' in metrics:
                        valid_metrics[pt] = metrics
                
                if not valid_metrics:
                    continue
                    
                 
                entropy_scores = {pt: metrics.get('entropy', float('inf')) 
                                for pt, metrics in valid_metrics.items()}
                if entropy_scores:
                    sharpest_prompt = min(entropy_scores.items(), key=lambda x: x[1])
                    
                     
                    gini_scores = {pt: metrics.get('gini_coefficient', 0) 
                                  for pt, metrics in valid_metrics.items()}
                    most_uneven_prompt = max(gini_scores.items(), key=lambda x: x[1])
                    
                    finding = {
                        "feature": feature_name,
                        "dataset": dataset_name,
                        "sharpest_by_entropy": {
                            "prompt": sharpest_prompt[0],
                            "entropy": sharpest_prompt[1],
                            "interpretation": f"数据集 {dataset_name} 中 {sharpest_prompt[0]} 显示最低的信息熵，分布最集中"
                        },
                        "most_uneven_by_gini": {
                            "prompt": most_uneven_prompt[0],
                            "gini": most_uneven_prompt[1],
                            "interpretation": f"数据集 {dataset_name} 中 {most_uneven_prompt[0]} 显示最高的基尼系数，分布最不均匀"
                        }
                    }
                    
                    report["key_findings"].append(finding)
        
         
        with open(f"{self.output_dir}/real_data_analysis_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
         
        self._print_analysis_summary_real(report)
    
    def _print_analysis_summary_real(self, report):
        """打印基于真实数据的分析摘要"""
        print("\n" + "="*60)
        print("真实数据分布分析结果摘要")
        print("="*60)
        
        for finding in report["key_findings"]:
            feature = finding["feature"]
            dataset = finding["dataset"]
            sharpest = finding["sharpest_by_entropy"]
            most_uneven = finding["most_uneven_by_gini"]
            
            print(f"\n特征: {feature} | 数据集: {dataset}")
            print(f"  最尖锐分布: {sharpest['interpretation']}")
            print(f"  最不均匀分布: {most_uneven['interpretation']}")
        
        print(f"\n结果文件已保存到: {self.output_dir}")
        print("请查看生成的图表和报告文件以获取详细信息。")

def main():
    """主函数"""
    print("LLM输出分布尖锐度分析工具（基于真实数据）")
    print("="*50)
    
    analyzer = DistributionAnalyzer()
    analyzer.run_distribution_experiments_from_real_data()

if __name__ == "__main__":
    main() 