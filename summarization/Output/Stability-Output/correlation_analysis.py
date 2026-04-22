#!/usr/bin/env python3
 
"""
计算两个stability score文件之间的相关性分析
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

 
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class CorrelationAnalyzer:
    def __init__(self, ground_truth_file, processed_file):
        self.ground_truth_file = ground_truth_file
        self.processed_file = processed_file
        self.ground_truth_data = None
        self.processed_data = None
        
    def load_data(self):
        """加载两个JSON文件"""
        print("正在加载数据文件...")
        
        try:
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                self.ground_truth_data = json.load(f)
            print(f"✓ 成功加载ground truth文件: {len(self.ground_truth_data)} 条记录")
            
            with open(self.processed_file, 'r', encoding='utf-8') as f:
                self.processed_data = json.load(f)
            print(f"✓ 成功加载处理后文件: {len(self.processed_data)} 条记录")
            
        except Exception as e:
            print(f"❌ 加载文件失败: {e}")
            return False
        
        return True
    
    def extract_overall_scores(self):
        """提取每个数据集+查询的总体stability_score"""
        gt_scores = []
        proc_scores = []
        identifiers = []
        
         
        for i, (gt_item, proc_item) in enumerate(zip(self.ground_truth_data, self.processed_data)):
             
            gt_id = f"{gt_item.get('dataset', 'unknown')}_{gt_item.get('query', 'unknown')}"
            proc_id = f"{proc_item.get('dataset', 'unknown')}_{proc_item.get('query', 'unknown')}"
            
            if gt_id != proc_id:
                print(f"⚠️  警告: 记录不匹配 - GT: {gt_id}, Processed: {proc_id}")
                continue
            
            gt_score = gt_item.get('stability_score')
            proc_score = proc_item.get('stability_score')
            
            if gt_score is not None and proc_score is not None:
                gt_scores.append(float(gt_score))
                proc_scores.append(float(proc_score))
                identifiers.append(gt_id)
        
        return np.array(gt_scores), np.array(proc_scores), identifiers
    
    def extract_pair_scores(self):
        """提取所有配对的stability_score"""
        gt_scores = []
        proc_scores = []
        pair_identifiers = []
        
        for gt_item, proc_item in zip(self.ground_truth_data, self.processed_data):
            dataset_id = f"{gt_item.get('dataset', 'unknown')}_{gt_item.get('query', 'unknown')}"
            
            gt_pairs = gt_item.get('pair_details', [])
            proc_pairs = proc_item.get('pair_details', [])
            
             
            for j, (gt_pair, proc_pair) in enumerate(zip(gt_pairs, proc_pairs)):
                gt_score = gt_pair.get('stability_score')
                proc_score = proc_pair.get('stability_score')
                
                if gt_score is not None and proc_score is not None:
                    gt_scores.append(float(gt_score))
                    proc_scores.append(float(proc_score))
                    pair_id = f"{dataset_id}_pair_{j+1}"
                    pair_identifiers.append(pair_id)
        
        return np.array(gt_scores), np.array(proc_scores), pair_identifiers
    
    def calculate_correlations(self, gt_scores, proc_scores, score_type="Overall"):
        """计算相关性统计"""
        print(f"\n📊 {score_type} Stability Scores 相关性分析")
        print("=" * 50)
        
         
        print(f"数据点数量: {len(gt_scores)}")
        print(f"Ground Truth - 均值: {np.mean(gt_scores):.4f}, 标准差: {np.std(gt_scores):.4f}")
        print(f"Processed    - 均值: {np.mean(proc_scores):.4f}, 标准差: {np.std(proc_scores):.4f}")
        
         
        pearson_corr, pearson_p = pearsonr(gt_scores, proc_scores)
        spearman_corr, spearman_p = spearmanr(gt_scores, proc_scores)
        
        print(f"\n🔗 相关性分析:")
        print(f"皮尔逊相关系数: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
        print(f"斯皮尔曼相关系数: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
        
         
        mse = mean_squared_error(gt_scores, proc_scores)
        mae = mean_absolute_error(gt_scores, proc_scores)
        rmse = np.sqrt(mse)
        
        print(f"\n📏 误差分析:")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        
         
        differences = proc_scores - gt_scores
        print(f"\n📈 差异分析:")
        print(f"平均差异: {np.mean(differences):.4f}")
        print(f"差异标准差: {np.std(differences):.4f}")
        print(f"最大正差异: {np.max(differences):.4f}")
        print(f"最大负差异: {np.min(differences):.4f}")
        
        return {
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mean_diff': np.mean(differences),
            'std_diff': np.std(differences)
        }
    
    def create_visualizations(self, gt_scores, proc_scores, score_type="Overall"):
        """创建可视化图表"""
        print(f"\n🎨 正在生成 {score_type} 可视化图表...")
        
         
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{score_type} Stability Scores 相关性分析', fontsize=16, fontweight='bold')
        
         
        axes[0, 0].scatter(gt_scores, proc_scores, alpha=0.6, s=50, color='steelblue')
        
         
        min_score = min(min(gt_scores), min(proc_scores))
        max_score = max(max(gt_scores), max(proc_scores))
        axes[0, 0].plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.8, label='完美相关 (y=x)')
        
         
        z = np.polyfit(gt_scores, proc_scores, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(gt_scores, p(gt_scores), 'g-', alpha=0.8, label=f'回归线 (y={z[0]:.3f}x+{z[1]:.3f})')
        
        axes[0, 0].set_xlabel('Ground Truth Stability Score')
        axes[0, 0].set_ylabel('Processed Stability Score')
        axes[0, 0].set_title('散点图与回归分析')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
         
        residuals = proc_scores - gt_scores
        axes[0, 1].scatter(gt_scores, residuals, alpha=0.6, s=50, color='orange')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('Ground Truth Stability Score')
        axes[0, 1].set_ylabel('残差 (Processed - Ground Truth)')
        axes[0, 1].set_title('残差分析')
        axes[0, 1].grid(True, alpha=0.3)
        
         
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].axvline(x=np.mean(residuals), color='g', linestyle='-', alpha=0.8, 
                          label=f'均值: {np.mean(residuals):.3f}')
        axes[1, 0].set_xlabel('差异值 (Processed - Ground Truth)')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('差异分布直方图')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
         
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('残差正态性Q-Q图')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
         
        output_file = f'stability_correlation_analysis_{score_type.lower()}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存: {output_file}")
        
        plt.show()
        
        return output_file
    
    def generate_detailed_report(self, overall_stats, pair_stats):
        """生成详细的分析报告"""
        report = f"""
 

# 
- Ground Truth文件: {self.ground_truth_file}
- 处理后文件: {self.processed_file}
- 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

# 
- **皮尔逊相关系数**: {overall_stats['pearson_corr']:.4f} (p={overall_stats['pearson_p']:.4e})
- **斯皮尔曼相关系数**: {overall_stats['spearman_corr']:.4f} (p={overall_stats['spearman_p']:.4e})
- **均方根误差 (RMSE)**: {overall_stats['rmse']:.4f}
- **平均绝对误差 (MAE)**: {overall_stats['mae']:.4f}
- **平均差异**: {overall_stats['mean_diff']:.4f}

# 
- **皮尔逊相关系数**: {pair_stats['pearson_corr']:.4f} (p={pair_stats['pearson_p']:.4e})
- **斯皮尔曼相关系数**: {pair_stats['spearman_corr']:.4f} (p={pair_stats['spearman_p']:.4e})
- **均方根误差 (RMSE)**: {pair_stats['rmse']:.4f}
- **平均绝对误差 (MAE)**: {pair_stats['mae']:.4f}
- **平均差异**: {pair_stats['mean_diff']:.4f}

# 

## 
"""
        
         
        def assess_correlation(corr):
            if abs(corr) >= 0.9:
                return "非常强"
            elif abs(corr) >= 0.7:
                return "强"
            elif abs(corr) >= 0.5:
                return "中等"
            elif abs(corr) >= 0.3:
                return "弱"
            else:
                return "很弱"
        
        overall_strength = assess_correlation(overall_stats['pearson_corr'])
        pair_strength = assess_correlation(pair_stats['pearson_corr'])
        
        report += f"""
- **总体级别**: {overall_strength}相关性 (r={overall_stats['pearson_corr']:.3f})
- **配对级别**: {pair_strength}相关性 (r={pair_stats['pearson_corr']:.3f})

## 
- 总体级别的RMSE ({overall_stats['rmse']:.4f}) {'较小' if overall_stats['rmse'] < 1.0 else '较大'}
- 配对级别的RMSE ({pair_stats['rmse']:.4f}) {'较小' if pair_stats['rmse'] < 1.0 else '较大'}

## 
- 总体级别平均差异: {overall_stats['mean_diff']:.4f} ({'处理后分数偏高' if overall_stats['mean_diff'] > 0 else '处理后分数偏低' if overall_stats['mean_diff'] < 0 else '无明显偏差'})
- 配对级别平均差异: {pair_stats['mean_diff']:.4f} ({'处理后分数偏高' if pair_stats['mean_diff'] > 0 else '处理后分数偏低' if pair_stats['mean_diff'] < 0 else '无明显偏差'})
"""
        
         
        with open('stability_correlation_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✓ 详细报告已保存: stability_correlation_report.md")
        
        return report
    
    def run_analysis(self):
        """运行完整的相关性分析"""
        print("🔬 开始Stability Score相关性分析")
        print("=" * 60)
        
         
        if not self.load_data():
            return
        
         
        gt_overall, proc_overall, overall_ids = self.extract_overall_scores()
        overall_stats = self.calculate_correlations(gt_overall, proc_overall, "Overall")
        
         
        gt_pairs, proc_pairs, pair_ids = self.extract_pair_scores()
        pair_stats = self.calculate_correlations(gt_pairs, proc_pairs, "Pair")
        
         
        self.create_visualizations(gt_overall, proc_overall, "Overall")
        self.create_visualizations(gt_pairs, proc_pairs, "Pair")
        
         
        self.generate_detailed_report(overall_stats, pair_stats)
        
        print(f"\n🎉 分析完成! 共分析了 {len(gt_overall)} 个总体分数和 {len(gt_pairs)} 个配对分数")
        
        return overall_stats, pair_stats

if __name__ == "__main__":
     
    ground_truth_file = "Output/Stability-Output/cast_stability_score_result.json"
    processed_file = "Output/Stability-Output/human_cast_stability_score_result_Carly_refined.json"
    
     
    analyzer = CorrelationAnalyzer(ground_truth_file, processed_file)
    overall_stats, pair_stats = analyzer.run_analysis()