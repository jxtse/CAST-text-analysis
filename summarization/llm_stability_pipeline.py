# LLM Stability Evaluation Pipeline
# Evaluates summarization stability across multiple LLM providers and prompt types

import os
import json
import time
import re
import pandas as pd
import numpy as np
import math
from scipy import stats
from datetime import datetime
from openai import OpenAI
from google import genai

import os
import dotenv
dotenv.load_dotenv()

class Logger:
    """Handles logging to both console and file."""
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

class Dataset:
    """Represents a dataset with text items for processing."""
    def __init__(self, name, language, text_items):
        self.name = name
        self.language = language
        self.text_items = text_items

class Query:
    """Represents a user query with language specifications."""
    def __init__(self, query_text, query_language, output_language=None):
        self.query_text = query_text
        self.query_language = query_language
        self.output_language = output_language or query_language

class TextStructuredInsight:
    """Stores structured analysis results from LLM responses."""
    def __init__(self):
        self.general_response = ""
        self.bullet_list = []
        self.perspective = ""
        self.text_items = []
        self.user_query = ""
        self.output_language = ""

class LLMAPI:
    """Unified API client for multiple LLM providers."""
    def __init__(self):
        # Initialize SiliconFlow API client
        self.siliconflow_client = OpenAI(
            api_key=os.getenv("SiliconFlow_API_KEY"), 
            base_url="https://api.siliconflow.cn/v1",
            timeout=300.0
        )
        
        # Initialize Grok API client
        self.grok_client = OpenAI(
            api_key=os.getenv("Grok_API_KEY"),
            base_url="https://api.x.ai/v1",
            timeout=300.0
        )

        # Initialize Google API client
        self.gemini_client = genai.Client(api_key=os.getenv("Gemini_API_KEY"))

        # Initialize OpenRouter API client
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=300.0
        )
        
    async def text_analysis_with_nl_query(self, dataset, query, prompt_type="ap"):
        """Perform text analysis using specified LLM provider.
        
        Args:
            dataset: Dataset object containing text items
            query: Query object with user request
            prompt_type: Prompt type ("baseline", "ap", "tbs", "cast")
        """
        try:
            system_prompt = self.get_text_analysis_system_prompt(prompt_type)
            user_prompt = self.get_text_analysis_user_prompt(dataset.text_items, dataset.name, query.query_text, query.query_language, query.output_language)
            
            # 打印提示词信息，帮助调试
            print(f"System prompt length: {len(system_prompt)}")
            print(f"User prompt sample: {user_prompt[:100]}...")
            llm_type = "openai/gpt-4.1"
            response = await self.llm_requests_async(system_prompt, user_prompt, llm_type=llm_type)
            if llm_type in ["gemini-2.5-pro-preview-03-25", "gemini-2.5-flash-preview-04-17"]:
                content = response.text
            else:
                content = response.choices[0].message.content
            
            # 打印API返回的原始内容
            print(f"API返回内容样本: {content[:200]}...")
            
            # 创建一个默认的insight对象
            insight = TextStructuredInsight()
            insight.text_items = dataset.text_items
            insight.user_query = query.query_text
            insight.output_language = query.output_language
            
            # 更健壮的JSON解析
            try:
                # 尝试直接解析整个内容为JSON
                data = json.loads(content)
                
                # 处理baseline格式
                if prompt_type == "baseline" and "topic_identification_result" in data:
                    topics = data.get("topic_identification_result", [])
                    # 确保general_response不为空
                    insight.general_response = json.dumps(topics) if topics else content
                    # 将topics转换为bullet points
                    bullet_points = []
                    for topic in topics:
                        bullet_points.append({
                            "Title": topic.get("title", ""),
                            "Description": topic.get("description", "")
                        })
                    insight.bullet_list = bullet_points
                elif "summary" in data:
                    # AP格式
                    insight.general_response = data.get("summary", "")
                    insight.bullet_list = data.get("bullet_points", [])
                elif "Results" in data:
                    # TbS/CAST格式
                    results = data.get("Results", [])
                    insight.general_response = json.dumps(results) if results else content
                    insight.bullet_list = results
                else:
                    # 默认情况下，使用整个JSON内容
                    insight.general_response = content
                    # 尝试从JSON中提取所有可能是bullet points的部分
                    all_possible_bullet_points = []
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            if all(isinstance(item, dict) for item in value):
                                all_possible_bullet_points = value
                                break
                    
                    if all_possible_bullet_points:
                        insight.bullet_list = all_possible_bullet_points
                
                # 确保bullet_list不为空
                if not insight.bullet_list:
                    # 尝试从内容中提取结构
                    insight.bullet_list = self._extract_bullet_points(content)
                
                # 确保general_response不为空
                if not insight.general_response:
                    insight.general_response = content
                
                print(f"成功解析为JSON，bullet_points数量: {len(insight.bullet_list)}")
                
                return insight
            except json.JSONDecodeError:
                # 如果整个内容不是有效的JSON，尝试提取JSON部分
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    try:
                        response_json = content[json_start:json_end]
                        response_data = json.loads(response_json)
                        
                        # 处理不同格式的JSON
                        if prompt_type == "baseline" and "topic_identification_result" in response_data:
                            topics = response_data.get("topic_identification_result", [])
                            insight.general_response = json.dumps(topics) if topics else content
                            bullet_points = []
                            for topic in topics:
                                bullet_points.append({
                                    "Title": topic.get("title", ""),
                                    "Description": topic.get("description", "")
                                })
                            insight.bullet_list = bullet_points
                        elif "summary" in response_data:
                            insight.general_response = response_data.get("summary", "")
                            insight.bullet_list = response_data.get("bullet_points", [])
                        elif "Results" in response_data:
                            results = response_data.get("Results", [])
                            insight.general_response = json.dumps(results) if results else content
                            insight.bullet_list = results
                        else:
                            # 默认使用整个JSON
                            insight.general_response = response_json
                            all_possible_bullet_points = []
                            for key, value in response_data.items():
                                if isinstance(value, list) and len(value) > 0:
                                    if all(isinstance(item, dict) for item in value):
                                        all_possible_bullet_points = value
                                        break
                            
                            if all_possible_bullet_points:
                                insight.bullet_list = all_possible_bullet_points
                        
                        # 确保bullet_list不为空
                        if not insight.bullet_list:
                            insight.bullet_list = self._extract_bullet_points(content)
                        
                        # 确保general_response不为空
                        if not insight.general_response:
                            insight.general_response = content
                        
                        print(f"成功解析为JSON（从内容中提取），bullet_points数量: {len(insight.bullet_list)}")
                        
                        return insight
                    except Exception as json_e:
                        print(f"JSON解析错误: {str(json_e)}, 内容: {response_json[:100]}...")
                
                # 如果没有找到有效的JSON格式，使用原始内容
                print(f"没有找到有效的JSON格式，使用原始内容")
                
                # 尝试从非结构化内容中提取要点
                insight.general_response = content
                insight.bullet_list = self._extract_bullet_points(content)
                
                print(f"从非结构化内容中提取bullet_points，数量: {len(insight.bullet_list)}")
                
                return insight
                
        except Exception as e:
            print(f"文本分析API调用错误: {str(e)}")
            # 提供更详细的错误信息
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            raise e
            
    def _extract_bullet_points(self, content):
        """从文本内容中提取可能的bullet points"""
        bullet_points = []
        
        # 尝试从非结构化内容中提取要点
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or 
                         line.startswith('*') or 
                         (len(line) > 2 and line[0].isdigit() and '.' in line[:3])):
                # 处理数字列表项，如"1. 内容"
                if len(line) > 2 and line[0].isdigit() and '.' in line[:3]:
                    content_part = line.split('.', 1)[1].strip()
                    bullet_points.append({
                        "Title": f"Point {len(bullet_points)+1}",
                        "Description": content_part
                    })
                else:
                    # 处理符号列表项，如"• 内容"或"- 内容"
                    content_part = line[1:].strip() if len(line) > 1 else line
                    bullet_points.append({
                        "Title": f"Point {len(bullet_points)+1}",
                        "Description": content_part
                    })
        
        # 如果未找到符号列表，尝试按段落分割
        if not bullet_points and len(lines) > 1:
            for i, line in enumerate(lines):
                if line.strip():
                    bullet_points.append({
                        "Title": f"Point {len(bullet_points)+1}",
                        "Description": line.strip()
                    })
                    # 最多提取5个点
                    if len(bullet_points) >= 5:
                        break
        
        # 如果仍未找到，至少返回一个点
        if not bullet_points:
            bullet_points.append({
                "Title": "Summary",
                "Description": content[:200] + ("..." if len(content) > 200 else "")
            })
        
        return bullet_points
    
    async def score_bullet_point(self, results_array):
        """评估子弹点列表的稳定性，类似于C#中的ScoreBulletPoint方法"""
        try:
            # 按dataset和query分组数据
            grouped_data = self._group_results_by_dataset_query(results_array)
            all_scores = []
            
            # 对每个数据集-查询对进行稳定性评估
            for key, group_info in list(grouped_data.items()):
                dataset_name = group_info["dataset"]
                query_text = group_info["query"]
                
                # 获取这个数据集-查询对的所有轮次的bullet points
                all_generations = group_info["generations"]
                num_generations = len(all_generations)
                print(f"评估 '{dataset_name}' - '{query_text}'，有 {num_generations} 轮生成")
                
                # 如果生成次数小于2，跳过评估
                if num_generations < 2:
                    print(f"  跳过评估 - 生成次数不足")
                    continue
                
                # 考虑所有可能的轮次组合进行评估
                pair_scores = []
                # 计算需要评估的组合总数
                total_pairs = (num_generations * (num_generations - 1)) // 2
                pair_count = 0
                
                # 遍历所有可能的轮次组合
                for i in range(num_generations - 1):
                    for j in range(i + 1, num_generations):
                        pair_count += 1
                        print(f"  评估组合 {pair_count}/{total_pairs}: 轮次 {i+1} vs 轮次 {j+1}")
                        
                        bullet_group1 = self._convert_to_bullet_point_group(all_generations[i])
                        bullet_group2 = self._convert_to_bullet_point_group(all_generations[j])
                        
                        # 如果任一组为空，跳过此组合
                        if len(bullet_group1) == 0 or len(bullet_group2) == 0:
                            print(f"    跳过 - 某一组bullet points为空")
                            continue
                        
                        # 准备评分提示
                        system_prompt = self.get_bullet_point_stability_system_prompt()
                        
                        # 构建JSON输入
                        bullet_point_data = {
                            "BulletPointGroup1": bullet_group1,
                            "BulletPointGroup2": bullet_group2
                        }
                        
                        # 转换为JSON字符串
                        user_prompt = json.dumps(bullet_point_data, ensure_ascii=False, indent=2)
                        print(f"    发送评估请求，提示词长度: {len(user_prompt)}")
                        
                        # 调用API进行语义匹配和评分
                        try:
                            response = await self.llm_requests_async(system_prompt, user_prompt, llm_type="openai/gpt-4.1")
                            content = response.choices[0].message.content
                            print(f"    收到API响应，长度: {len(content)}")
                            
                            # 尝试提取JSON部分
                            json_content = ""
                            try:
                                # 先尝试直接解析整个内容
                                match_result = json.loads(content)
                                json_content = content
                            except json.JSONDecodeError:
                                # 如果整个内容不是有效的JSON，提取JSON部分
                                json_start = content.find('{')
                                json_end = content.rfind('}') + 1
                                
                                if json_start >= 0 and json_end > json_start:
                                    json_content = content[json_start:json_end]
                                    try:
                                        match_result = json.loads(json_content)
                                    except Exception as json_e:
                                        print(f"    JSON解析错误: {str(json_e)}")
                                        continue
                                else:
                                    print(f"    未找到有效的JSON，跳过")
                                    continue
                            
                            # 从匹配结果中提取必要信息
                            semantic_matches = match_result.get("SemanticMatches", [])
                            matched_positions = match_result.get("MatchedPositions", {})
                            analysis_details = match_result.get("AnalysisDetails", "")
                            
                            # 后处理计算SemanticScore和MatchRatio
                            # 1. 计算SemanticScore (所有匹配项的平均语义相似度)
                            total_similarity = sum(match.get("SimilarityScore", 0) for match in semantic_matches)
                            semantic_score = total_similarity / len(semantic_matches) if semantic_matches else 0
                            
                            # 2. 计算匹配率 - 使用多种度量
                            # 原始匹配率(基于较小组)
                            min_group_size = min(len(bullet_group1), len(bullet_group2))
                            match_ratio_min = len(semantic_matches) / min_group_size if min_group_size > 0 else 0
                            
                            # Jaccard系数: 交集大小/并集大小
                            union_size = len(bullet_group1) + len(bullet_group2) - len(semantic_matches)
                            jaccard_index = len(semantic_matches) / union_size if union_size > 0 else 0
                            
                            # 平均匹配率: 匹配项数/两组平均大小
                            avg_size = (len(bullet_group1) + len(bullet_group2)) / 2
                            avg_match_ratio = len(semantic_matches) / avg_size if avg_size > 0 else 0
                            
                            # 3. 计算未匹配惩罚分数 
                            # 使用 (1 - Jaccard系数) 作为主要未匹配指标
                            unmatch_ratio = 1 - jaccard_index
                            
                            # 差异率 = |group1.size - group2.size| / max(group1.size, group2.size)
                            size_difference = abs(len(bullet_group1) - len(bullet_group2)) / max(len(bullet_group1), len(bullet_group2)) if max(len(bullet_group1), len(bullet_group2)) > 0 else 0
                            
                            # 组合惩罚分数 (0-1范围)
                            # 增加未匹配率的权重以反映其重要性
                            penalty_score = (unmatch_ratio * 0.8 + size_difference * 0.2)
                            # 应用轻微非线性变换，使高惩罚更显著，低惩罚影响更小
                            # penalty_score = math.pow(penalty_score, 0.8)  # 幂小于1使曲线更陡峭
                            
                            print(f"    计算得到 SemanticScore: {semantic_score:.2f}, Jaccard指数: {jaccard_index:.2f}, 未匹配惩罚: {penalty_score:.2f}")
                            print(f"    原始匹配率: {match_ratio_min:.2f}, 平均匹配率: {avg_match_ratio:.2f}, 大小差异: {size_difference:.2f}")
                            
                            # 使用LLM提供的匹配位置计算Kendall tau
                            kendall_tau, p_value, position_consistency = self._calculate_kendall_tau_from_matched_positions(matched_positions)
                            print(f"    计算Kendall tau相关系数: {kendall_tau:.4f}, p值: {p_value:.4f}, 位置一致性: {position_consistency:.4f}")
                            
                            # 计算最终稳定性分数
                            # 主要分数权重：语义相似性占60%，位置一致性占40%
                            semantic_weight = 0.6
                            position_weight = 0.4
                            
                            # 归一化语义分数 (0-5 => 0-1)
                            normalized_semantic_score = semantic_score / 5
                            
                            # 计算内容一致性分数 (0-1范围)
                            content_score = (normalized_semantic_score * semantic_weight) + (position_consistency * position_weight)
                            
                            # 整合未匹配惩罚 - 使用乘法形式捕捉"短板效应"
                            # 防止过度惩罚，重新缩放惩罚分数
                            # penalty_factor 范围：[0.5, 1]，0.5表示最大惩罚，1表示无惩罚
                            # 这确保了即使惩罚最大，内容分数也至少保留50%
                            max_penalty_effect = 0.5  # 控制惩罚对最终分数的最大影响
                            penalty_factor = 1 - (penalty_score * max_penalty_effect)
                            
                            # 最终分数 = 内容分数 * 惩罚因子 (乘法形式)
                            total_score = content_score * penalty_factor
                            
                            # 确保分数在0-1范围内
                            total_score = max(0, min(1, total_score))
                            
                            # 转换为10分制
                            stability_score = total_score * 10
                            
                            # 归一化其他分数为10分制
                            semantic_score_10 = normalized_semantic_score * 10
                            position_score_10 = position_consistency * 10
                            jaccard_index_10 = jaccard_index * 10
                            penalty_score_10 = penalty_score * 10
                            
                            # 创建稳定性结果
                            pair_result = {
                                "dataset": dataset_name,
                                "query": query_text,
                                "round_pair": f"{i+1}-{j+1}",
                                "stability_score": stability_score,
                                "semantic_score": semantic_score_10,
                                "position_score": position_score_10,
                                "jaccard_index": jaccard_index_10,
                                "original_match_ratio": match_ratio_min * 10,
                                "average_match_ratio": avg_match_ratio * 10,
                                "penalty_score": penalty_score_10,
                                "penalty_factor": penalty_factor,
                                "kendall_tau": kendall_tau,
                                "kendall_p_value": p_value,
                                "matched_items_count": len(semantic_matches),
                                "group1_count": len(bullet_group1),
                                "group2_count": len(bullet_group2),
                                "size_difference": size_difference,
                                "semantic_matches": semantic_matches,
                                "matched_positions": matched_positions,
                                "analysis_details": analysis_details
                            }
                            
                            pair_scores.append(pair_result)
                            print(f"    评分完成，稳定性分数: {stability_score:.2f}/10")
                            print(f"    语义分数: {semantic_score_10:.2f}/10, 位置分数: {position_score_10:.2f}/10, 匹配比例: {avg_match_ratio:.2f}/10")
                            
                        except Exception as api_e:
                            print(f"    API调用失败: {str(api_e)}")
                            continue
                
                # 如果有成功评估的组合
                if pair_scores:
                    # 计算平均分数
                    avg_stability = sum(item["stability_score"] for item in pair_scores) / len(pair_scores)
                    avg_semantic = sum(item["semantic_score"] for item in pair_scores) / len(pair_scores)
                    avg_position = sum(item["position_score"] for item in pair_scores) / len(pair_scores)
                    avg_match_ratio = sum(item["average_match_ratio"] for item in pair_scores) / len(pair_scores)
                    
                    # 创建汇总结果
                    summary_result = {
                        "dataset": dataset_name,
                        "query": query_text,
                        "num_generations": num_generations,
                        "num_evaluated_pairs": len(pair_scores),
                        "stability_score": avg_stability,
                        "semantic_score": avg_semantic,
                        "position_score": avg_position,
                        "match_ratio": avg_match_ratio,
                        "pair_details": pair_scores
                    }
                    
                    all_scores.append(summary_result)
                    print(f"  汇总评分完成，平均稳定性分数: {avg_stability:.2f}/10")
                    print(f"  平均语义分数: {avg_semantic:.2f}/10, 平均位置分数: {avg_position:.2f}/10, 平均匹配比例: {avg_match_ratio:.2f}/10")
            
            # 保存所有评分结果
            if all_scores:
                with open("Output/Stability-Output/bullet_point_stability_score.json", "w", encoding="utf-8") as f:
                    json.dump(all_scores, f, indent=2, ensure_ascii=False)
                
                # 生成一个简化版本的结果（不包含详细信息）
                simplified_scores = []
                for score in all_scores:
                    simplified_score = {k: v for k, v in score.items() if k != "pair_details"}
                    simplified_scores.append(simplified_score)
                    
                with open("Output/Stability-Output/bullet_point_stability_score_summary.json", "w", encoding="utf-8") as f:
                    json.dump(simplified_scores, f, indent=2, ensure_ascii=False)
                    
                print(f"已保存 {len(all_scores)} 个评分结果")
            else:
                print("没有评分结果可保存")
            
            return all_scores
                
        except Exception as e:
            print(f"评分过程中出错: {str(e)}")
            raise e
            
    async def llm_requests_async(self, system_prompt, user_prompt, llm_type, seed=42):
        """LLM API请求"""
        try:
            if llm_type in ["gpt-4o", "o4-mini", "gpt-4.1", "openai/o4-mini", "openai/gpt-4.1"]:
                response = self.openai_client.chat.completions.create(
                    model=llm_type,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    # max_tokens=max_tokens,
                    # temperature=0,
                    # top_p=1,
                    # n=1,
                    # seed=seed
                )
            elif llm_type in ["grok-3-beta"]:
                response = self.grok_client.chat.completions.create(
                    model=llm_type,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}    
                    ],
                    temperature=0,
                    top_p=1,
                    n=1,
                    seed=seed   
                )
            elif llm_type in ["Pro/deepseek-ai/DeepSeek-R1", "Qwen/Qwen3-235B-A22B"]:
                response = self.siliconflow_client.chat.completions.create(
                    model=llm_type,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    # max_tokens=max_tokens,
                    temperature=0,
                    top_p=1,
                    n=1,
                    seed=seed   
                )
            elif llm_type in ["gemini-2.5-pro-preview-03-25", "gemini-2.5-flash-preview-04-17"]:
                response = self.gemini_client.models.generate_content(
                    model=llm_type,
                    contents=f"""
                    <system_prompt> {system_prompt} </system_prompt>
                    
                    <user_prompt> {user_prompt} </user_prompt>
                    """,
                    # max_tokens=max_tokens,
                    # temperature=0,
                    # top_p=1,
                    # n=1,
                    # seed=seed
                )
            else:
                raise ValueError(f"不支持的LLM类型: {llm_type}")
            
            return response
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            raise e
    
    def get_text_analysis_system_prompt(self, prompt_type="ap"):
        """获取文本分析系统提示词
        
        Args:
            prompt_type: 提示词类型，可选值为 "baseline", "ap", "tbs", "cast"
        """
        try:
            prompt_files = {
                "baseline": "baseline_prompt.md",
                "ap": "ap_prompt.md",
                "tbs": "tbs_prompt.md",
                "cast": "cast_prompt.md"
            }
            
            file_name = prompt_files.get(prompt_type.lower(), "ap_prompt.md")
            print(f"正在加载提示词文件: {file_name}")
            
            with open(file_name, "r", encoding="utf-8") as f:
                prompt = f.read()
            return prompt
        except Exception as e:
            print(f"读取文本分析提示词文件出错: {str(e)}")
            # 如果文件读取失败，尝试读取默认文件
            try:
                with open("ap_prompt.md", "r", encoding="utf-8") as f:
                    prompt = f.read()
                return prompt
            except Exception as e2:
                print(f"读取默认提示词文件也失败: {str(e2)}")
                return ""
    
    def get_text_analysis_user_prompt(self, text_items, column_name, query_text, query_language, output_language):
        """生成文本分析用户提示词"""
        prompt = f"User Query: {query_text}\n"
        prompt += f"Query Language: {query_language}\n"
        
        if output_language:
            prompt += f"Output Language: {output_language}\n"
            
        if column_name:
            prompt += f"Column Name: {column_name}\n"
            
        prompt += "Text Items:\n"
        
        for i, item in enumerate(text_items):
            prompt += f"[{i+1}] {item}\n"
            
        return prompt
    
    def get_bullet_point_stability_system_prompt(self):
        """获取子弹点稳定性评估系统提示词"""
        try:
            with open("stability_evaluation_prompt.md", "r", encoding="utf-8") as f:
                prompt = f.read()
            return prompt
        except Exception as e:
            print(f"读取稳定性评估提示词文件出错: {str(e)}")
    
    def get_bullet_point_stability_user_prompt(self, results_array):
        """生成子弹点稳定性评估用户提示词"""
        # 确保我们有数据要处理
        if not results_array or len(results_array) == 0:
            print("警告: results_array为空，无法生成稳定性评估提示")
            return json.dumps({"BulletPointGroup1": [], "BulletPointGroup2": []})
            
        print(f"处理results_array，包含{len(results_array)}组结果")
        
        # 按dataset和query分组数据
        grouped_data = self._group_results_by_dataset_query(results_array)
        print(f"分组后有{len(grouped_data)}个数据集-查询对")
        
        # 我们只需要选择第一个有足够数据的分组进行评估
        for key, group_info in list(grouped_data.items()):
            # 获取这个数据集-查询对的所有轮次的bullet points
            all_generations = group_info["generations"]
            print(f"数据集-查询对 '{key}' 有 {len(all_generations)} 轮生成")
            
            # 如果生成次数至少有2次，可以进行稳定性评估
            if len(all_generations) >= 2:
                # 选择前两轮生成结果进行比较
                bullet_group1 = self._convert_to_bullet_point_group(all_generations[0])
                bullet_group2 = self._convert_to_bullet_point_group(all_generations[1])
                
                print(f"第一组bullet points: {len(bullet_group1)}个")
                print(f"第二组bullet points: {len(bullet_group2)}个")
                
                # 确保两组都有内容
                if len(bullet_group1) == 0 or len(bullet_group2) == 0:
                    print(f"警告: 数据集-查询对 '{key}' 的bullet points为空，跳过")
                    continue
                
                # 构建JSON输入
                bullet_point_data = {
                    "BulletPointGroup1": bullet_group1,
                    "BulletPointGroup2": bullet_group2
                }
                
                # 转换成JSON
                user_prompt = json.dumps(bullet_point_data, ensure_ascii=False, indent=2)
                print(f"创建user_prompt (长度: {len(user_prompt)})")
                return user_prompt
        
        # 如果没有可以评估的数据
        print("警告: 没有找到可评估的数据，返回空结果")
        return json.dumps({"BulletPointGroup1": [], "BulletPointGroup2": []})
        
    def _group_results_by_dataset_query(self, results_array):
        """将结果按dataset和query分组"""
        grouped_data = {}
        
        try:
            for group in results_array:
                if isinstance(group, list):
                    for result in group:
                        if isinstance(result, dict):
                            dataset = result.get("Dataset", "")
                            query = result.get("Query", "")
                            key = f"{dataset}|{query}"
                            
                            if key not in grouped_data:
                                grouped_data[key] = {
                                    "dataset": dataset,
                                    "query": query,
                                    "generations": []
                                }
                            
                            bullet_points = result.get("BulletPoint", [])
                            if bullet_points:
                                grouped_data[key]["generations"].append(bullet_points)
        except Exception as e:
            print(f"分组数据时出错: {str(e)}")
        
        return grouped_data
        
    def _convert_to_bullet_point_group(self, bullet_points):
        """将bullet points转换为评估需要的格式"""
        formatted_bullet_points = []
        
        try:
            if not bullet_points:
                print("警告: bullet_points为空")
                return formatted_bullet_points
                
            for i, point in enumerate(bullet_points):
                # 打印出当前处理的point类型和内容
                print(f"处理第{i+1}个bullet point，类型: {type(point)}")
                
                # 如果是dict，确保它有需要的字段
                if isinstance(point, dict):
                    bullet_point = point.copy()
                    
                    # 确保有Title和Description字段
                    if "Title" not in bullet_point and "title" in bullet_point:
                        bullet_point["Title"] = bullet_point["title"]
                    if "Description" not in bullet_point and "description" in bullet_point:
                        bullet_point["Description"] = bullet_point["description"]
                    
                    # 如果还是没有Title，使用第一个找到的字符串值
                    if "Title" not in bullet_point:
                        for k, v in bullet_point.items():
                            if isinstance(v, str) and k != "Description":
                                bullet_point["Title"] = v
                                break
                    
                    # 如果没有找到字符串值作为标题，使用键名
                    if "Title" not in bullet_point:
                        bullet_point["Title"] = list(bullet_point.keys())[0] if bullet_point else "Point"
                    
                    # 确保有Position字段
                    if "Position" not in bullet_point:
                        bullet_point["Position"] = i
                    
                    formatted_bullet_points.append(bullet_point)
                elif isinstance(point, str):
                    # 如果是字符串，尝试分离标题和描述
                    title_desc = self._extract_title_description(point)
                    formatted_bullet_points.append({
                        "Title": title_desc["title"],
                        "Description": title_desc["description"],
                        "Position": i
                    })
                else:
                    # 其他类型，转换为字符串
                    point_str = str(point)
                    title_desc = self._extract_title_description(point_str)
                    formatted_bullet_points.append({
                        "Title": title_desc["title"],
                        "Description": title_desc["description"],
                        "Position": i
                    })
        except Exception as e:
            print(f"转换bullet points格式时出错: {str(e)}")
        
        return formatted_bullet_points
    
    def _extract_title_description(self, bullet_point_text):
        """从bullet point文本中提取标题和描述"""
        # 尝试多种可能的分隔符
        separators = [": ", " - ", ". ", "、", "，", "。"]
        
        for sep in separators:
            if sep in bullet_point_text:
                parts = bullet_point_text.split(sep, 1)
                return {"title": parts[0].strip(), "description": parts[1].strip()}
        
        # 如果没有发现分隔符，将整个文本作为标题
        return {"title": bullet_point_text, "description": ""}
        
    def _calculate_kendall_tau_from_matched_positions(self, matched_positions):
        """根据LLM提供的匹配位置信息计算Kendall tau相关系数
        
        Args:
            matched_positions: 包含匹配位置的字典，格式为：
                               {"Group1Positions": [0, 2, 4], 
                                "Group2Positions": [1, 3, 5]}
                                
        Returns:
            tau: Kendall tau相关系数
            p_value: p值
            position_consistency: 位置一致性分数 (0-1)
        """
        try:
            # 提取匹配位置
            group1_positions = matched_positions.get("Group1Positions", [])
            group2_positions = matched_positions.get("Group2Positions", [])
            
            # 确保两个列表长度相同
            if len(group1_positions) != len(group2_positions) or len(group1_positions) < 2:
                print(f"匹配位置数据不完整或不足: G1={group1_positions}, G2={group2_positions}")
                return 0, 1.0, 0
                
            # 计算Kendall tau系数
            tau, p_value = stats.kendalltau(group1_positions, group2_positions)
            
            # 如果tau是NaN，返回0
            if math.isnan(tau):
                return 0, 1.0, 0
                
            # 计算位置一致性分数 (将tau从[-1,1]范围映射到[0,1]范围)
            position_consistency = (tau + 1) / 2
                
            return tau, p_value, position_consistency
            
        except Exception as e:
            print(f"根据匹配位置计算Kendall tau时出错: {str(e)}")
            return 0, 1.0, 0

class LLMStabilityPipeline:
    def __init__(self, log_path="Output/Stability-Output/log.txt"):
        directory_path = os.path.dirname(log_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
        self.logger = Logger(log_path)
        self.llm_api = LLMAPI()
        
        # 设置相对路径和绝对路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(base_dir, "Input/Stability-Input/xlsx/Multilingual Text Summarization Datasets.xlsx")
        self.query_path = os.path.join(base_dir, "Input/Stability-Input/xlsx/Multilingual NL Queries without Column Name.xlsx")
        
        # 如果文件不存在，使用样例数据
        if not os.path.exists(self.dataset_path) or not os.path.exists(self.query_path):
            self.logger.write_line(f"[{datetime.now()}] [WARNING] 找不到数据集或查询文件，将使用样例数据")
            self.use_sample_data = True
        else:
            self.use_sample_data = False
        
    async def run(self, prompt_types):
        """运行稳定性测试管道"""
        self.logger.write_line(f"[{datetime.now()}] [INFO] Starting LLM Stability Pipeline")
        
        # 创建输出目录
        stability_result_path = "Output/Stability-Output/stability_results.json"
        directory_path = os.path.dirname(stability_result_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        
        # 运行子弹点生成和评分稳定性测试
        await self.score_bullet_point(stability_result_path, prompt_types)
        
        self.logger.write_line(f"[{datetime.now()}] [INFO] LLM Stability Pipeline completed")
        self.logger.close()
        
    async def score_bullet_point(self, score_bullet_point_result, prompt_types):
        """评估子弹点生成的稳定性"""
        # 加载数据集和查询
        if self.use_sample_data:
            datasets = self.get_sample_datasets()
            queries = self.get_sample_queries()
        else:
            datasets = self.get_datasets_from_excel(self.dataset_path)
            queries = self.get_queries_from_excel(self.query_path)
        
        # 设置要测试的提示词类型
        round_times = 5  # 每个实例生成5次
        
        # 创建数据集和查询的组合
        instances = []
        for dataset in datasets:
            for query in queries:
                if dataset.language == query.query_language:
                    instances.append({"Dataset": dataset, "Query": query})
        
        self.logger.write_line(f"[{datetime.now()}] [INFO] Dataset Number: {len(instances)}")
        
        all_results = {}
        
        # 为每种prompt类型测试
        for prompt_type in prompt_types:
            self.logger.write_line(f"[{datetime.now()}] [INFO] 开始测试 {prompt_type} prompt")
            
            final_result = []
            all_results[prompt_type] = final_result
            
            for instance in instances:
                results = []
                final_result.append(results)
                
                for i in range(round_times):
                    while True:
                        try:
                            # 等待5秒避免请求限制
                            time.sleep(5)
                            
                            start_time = datetime.now()
                            analysis_result = await self.llm_api.text_analysis_with_nl_query(instance["Dataset"], instance["Query"], prompt_type)
                            end_time = datetime.now()
                            analysis_time = (end_time - start_time).total_seconds()
                            
                            time.sleep(5)
                            
                            if not analysis_result.general_response:
                                raise Exception("Analysis result is empty.")
                            
                            result = {
                                "Dataset": f"{instance['Dataset'].name}_{instance['Dataset'].language}",
                                "Query": instance["Query"].query_text,
                                "AnalysisResult": analysis_result.general_response,
                                "BulletPoint": analysis_result.bullet_list,
                                # "Perspective": analysis_result.perspective,
                                "Time": analysis_time,
                                "PromptType": prompt_type
                            }
                            
                            results.append(result)
                            
                            self.logger.write_line(
                                f"[{datetime.now()}] [INFO] PromptType: {prompt_type}, "
                                f"Dataset: {instance['Dataset'].name}, "
                                f"Language: {instance['Dataset'].language}, "
                                f"Query: {instance['Query'].query_text}, "
                                f"Time: {analysis_time}s"
                            )
                            
                            # 保存中间结果
                            with open(f"Output/Stability-Output/{prompt_type}_stability_results.json", "w", encoding="utf-8") as f:
                                json.dump(final_result, f, indent=2, ensure_ascii=False)
                                
                            break
                        except Exception as e:
                            self.logger.write_line(
                                f"[{datetime.now()}] [ERROR] PromptType: {prompt_type}, "
                                f"Dataset: {instance['Dataset'].name}, "
                                f"Language: {instance['Dataset'].language}, "
                                f"Query: {instance['Query'].query_text}, "
                                f"Error: {str(e)}"
                            )
                            
                            # 如果是限流错误，等待1分钟
                            if "Throttling" in str(e):
                                time.sleep(60)
                            else:
                                time.sleep(5)
            
            # 为当前prompt类型评分子弹点稳定性
            self.logger.write_line(f"[{datetime.now()}] [INFO] 开始评估 {prompt_type} prompt的稳定性")
            stability_score_result = await self.llm_api.score_bullet_point(final_result)
            
            # 保存评分结果
            with open(f"Output/Stability-Output/{prompt_type}_stability_score_result.json", "w", encoding="utf-8") as f:
                json.dump(stability_score_result, f, indent=2, ensure_ascii=False)
                
            self.logger.write_line(f"[{datetime.now()}] [INFO] {prompt_type} prompt的稳定性评估完成并保存")
            
        # 比较四种prompt的稳定性结果
        self.compare_stability_results(all_results, prompt_types)
    
    def score_bullet_point_from_file(self, file_path):
        """从文件中读取结果并评分"""
        import asyncio
        
        if not os.path.exists(file_path):
            self.logger.write_line(f"[{datetime.now()}] [ERROR] File not found: {file_path}")
            return
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                final_result = json.load(f)
                
            # 评分子弹点稳定性
            stability_score_result = asyncio.run(self.llm_api.score_bullet_point(final_result))
            
            # 保存评分结果
            output_path = "Output/Stability-Output/stability_score_from_file.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(stability_score_result, f, indent=2, ensure_ascii=False)
                
            # 生成一个简化版本的结果（不包含详细信息）
            simplified_scores = []
            for score in stability_score_result:
                simplified_score = {k: v for k, v in score.items() if k != "pair_details"}
                simplified_scores.append(simplified_score)
                
            with open("Output/Stability-Output/stability_score_from_file_summary.json", "w", encoding="utf-8") as f:
                json.dump(simplified_scores, f, indent=2, ensure_ascii=False)
                
            # 计算平均分和统计数据
            if stability_score_result:
                # 从汇总结果中提取统计信息
                avg_stability = sum(item.get("stability_score", 0) for item in stability_score_result) / len(stability_score_result)
                avg_semantic = sum(item.get("semantic_score", 0) for item in stability_score_result) / len(stability_score_result)
                avg_position = sum(item.get("position_score", 0) for item in stability_score_result) / len(stability_score_result)
                avg_jaccard = sum(item.get("jaccard_index", 0) for item in stability_score_result) / len(stability_score_result)
                avg_penalty = sum(item.get("penalty_score", 0) for item in stability_score_result) / len(stability_score_result)
                
                # 计算评估的总对数
                total_evaluated_pairs = sum(item.get("num_evaluated_pairs", 0) for item in stability_score_result)
                
                stats = {
                    "数据集-查询组合数": len(stability_score_result),
                    "总评估对数": total_evaluated_pairs,
                    "平均稳定性分数": f"{avg_stability:.2f}/10",
                    "平均语义一致性": f"{avg_semantic:.2f}/10",
                    "平均位置一致性": f"{avg_position:.2f}/10",
                    "平均Jaccard匹配指数": f"{avg_jaccard:.2f}/10",
                    "平均惩罚分数": f"{avg_penalty:.2f}/10"
                }
                
                # 保存统计数据
                with open("Output/Stability-Output/stability_score_stats.json", "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                
                self.logger.write_line(f"[{datetime.now()}] [INFO] 稳定性评估完成，共评估 {len(stability_score_result)} 个数据集-查询组合，{total_evaluated_pairs} 对生成结果")
                self.logger.write_line(f"[{datetime.now()}] [INFO] 平均稳定性分数: {avg_stability:.2f}/10")
                self.logger.write_line(f"[{datetime.now()}] [INFO] 平均语义一致性: {avg_semantic:.2f}/10")
                self.logger.write_line(f"[{datetime.now()}] [INFO] 平均位置一致性: {avg_position:.2f}/10")
                self.logger.write_line(f"[{datetime.now()}] [INFO] 平均Jaccard匹配指数: {avg_jaccard:.2f}/10")
                self.logger.write_line(f"[{datetime.now()}] [INFO] 平均惩罚分数: {avg_penalty:.2f}/10")
            else:
                self.logger.write_line(f"[{datetime.now()}] [WARNING] 未生成评分结果")
                
            self.logger.write_line(f"[{datetime.now()}] [INFO] 评分结果已保存至 {output_path}")
                
        except Exception as e:
            self.logger.write_line(f"[{datetime.now()}] [ERROR] 从文件评分失败: {str(e)}")
            print(f"错误详情: {str(e)}")  # 打印详细错误信息帮助调试
    
    def get_datasets_from_excel(self, path):
        """从Excel文件中读取数据集"""
        # 语言映射
        language_map = {
            "english": "en_US",
            "spanish": "es_ES",
            "french": "fr_FR",
            "german": "de_DE",
            "italian": "it_IT",
            "japanese": "ja_JP",
            "chinese": "zh_CN",
            # 可根据需要添加更多语言
        }
        
        datasets = []
        try:
            # 使用pandas读取Excel文件
            excel_file = pd.ExcelFile(path)
            
            for sheet_name in excel_file.sheet_names:
                # 解析工作表名称获取数据集名称和语言
                parts = sheet_name.split('_')
                if len(parts) >= 2:
                    name = parts[0]
                    language_key = parts[1].lower()
                    language = language_map.get(language_key, "en_US")
                    
                    # 读取工作表数据
                    df = excel_file.parse(sheet_name)
                    
                    # 最后一列是文本项
                    last_column = df.columns[-1]
                    text_items = df[last_column].dropna().tolist()
                    
                    # 从第二行开始（跳过标题行）
                    if len(text_items) > 0:
                        datasets.append(Dataset(name, language, text_items))
            
            return datasets
        except Exception as e:
            print(f"读取数据集时出错: {str(e)}")
            return []
    
    def get_datasets_from_benchmark(self, benchmark_name):
        """从基准测试中读取数据集"""
        # 这里应该根据实际情况实现
        # 简化实现，返回一个示例数据集
        if benchmark_name == "Yelp":
            # 创建示例Yelp数据集
            yelp_reviews = [
                "The food was amazing and the service was excellent.",
                "I waited for 30 minutes and the food was cold when it arrived.",
                "The ambiance is great but the prices are too high.",
                "Best restaurant in town, will definitely come back!",
                "The staff was rude and unhelpful."
            ]
            return [Dataset("Yelp", "en_US", yelp_reviews)]
        return []
    
    def get_queries_from_excel(self, path):
        """从Excel文件中读取查询"""
        # 语言映射
        language_map = {
            "english": "en_US",
            "spanish": "es_ES",
            "french": "fr_FR",
            "german": "de_DE",
            "italian": "it_IT",
            "japanese": "ja_JP",
            "chinese": "zh_CN",
            # 可根据需要添加更多语言
        }
        
        queries = []
        try:
            # 使用pandas读取Excel文件
            excel_file = pd.ExcelFile(path)
            
            for sheet_name in excel_file.sheet_names:
                # 解析工作表语言
                language = language_map.get(sheet_name.lower(), "en_US")
                
                # 读取工作表数据
                df = excel_file.parse(sheet_name)
                
                # 遍历行，从第二行开始（跳过标题行）
                for _, row in df.iloc[1:].iterrows():
                    # 检查前两列和第五列的查询文本
                    columns_to_check = [0, 1, 4]  # 对应于原代码中的列索引1、2和5
                    
                    for col_idx in columns_to_check:
                        if col_idx < len(df.columns):
                            query_text = row.iloc[col_idx]
                            if isinstance(query_text, str) and query_text.strip():
                                # 检查是否包含语言标签
                                match = re.search(r'<(.+?)>', query_text)
                                
                                if match:
                                    # 提取语言标签并从查询文本中移除
                                    lang_tag = match.group(1)
                                    query_text = re.sub(r'<.+?>', '', query_text).strip()
                                    output_language = language_map.get(lang_tag.lower(), None)
                                    
                                    queries.append(Query(query_text, language, output_language))
                                else:
                                    queries.append(Query(query_text, language))
            
            return queries
        except Exception as e:
            print(f"读取查询时出错: {str(e)}")
            return []

    def compare_stability_results(self, all_results, prompt_types):
        """比较不同prompt类型的稳定性结果
        
        Args:
            all_results: 包含所有prompt类型结果的字典
            prompt_types: prompt类型列表
        """
        try:
            self.logger.write_line(f"[{datetime.now()}] [INFO] 开始比较不同prompt类型的稳定性结果")
            
            # 加载每种prompt类型的稳定性评分结果
            stability_scores = {}
            for prompt_type in prompt_types:
                try:
                    with open(f"Output/Stability-Output/{prompt_type}_stability_score_result.json", "r", encoding="utf-8") as f:
                        stability_scores[prompt_type] = json.load(f)
                except Exception as e:
                    self.logger.write_line(f"[{datetime.now()}] [ERROR] 无法加载{prompt_type}的稳定性评分结果: {str(e)}")
                    stability_scores[prompt_type] = []
            
            # 比较各项指标
            comparison_result = {
                "prompt_comparison": {}
            }
            
            for prompt_type in prompt_types:
                scores = stability_scores[prompt_type]
                if not scores:
                    comparison_result["prompt_comparison"][prompt_type] = {
                        "average_stability_score": 0,
                        "average_semantic_score": 0,
                        "average_position_score": 0,
                        "average_match_ratio": 0,
                        "dataset_count": 0,
                        "pair_count": 0
                    }
                    continue
                
                # 计算平均分数
                total_stability = 0
                total_semantic = 0
                total_position = 0
                total_match_ratio = 0
                total_datasets = len(scores)
                total_pairs = 0
                
                for score in scores:
                    if "stability_score" in score:
                        total_stability += score.get("stability_score", 0)
                    if "semantic_score" in score:
                        total_semantic += score.get("semantic_score", 0)
                    if "position_score" in score:
                        total_position += score.get("position_score", 0)
                    if "match_ratio" in score:
                        total_match_ratio += score.get("match_ratio", 0)
                    total_pairs += score.get("num_evaluated_pairs", 0)
                
                # 计算平均值
                avg_stability = total_stability / total_datasets if total_datasets > 0 else 0
                avg_semantic = total_semantic / total_datasets if total_datasets > 0 else 0
                avg_position = total_position / total_datasets if total_datasets > 0 else 0
                avg_match_ratio = total_match_ratio / total_datasets if total_datasets > 0 else 0
                
                comparison_result["prompt_comparison"][prompt_type] = {
                    "average_stability_score": avg_stability,
                    "average_semantic_score": avg_semantic,
                    "average_position_score": avg_position,
                    "average_match_ratio": avg_match_ratio,
                    "dataset_count": total_datasets,
                    "pair_count": total_pairs
                }
            
            # 添加排名信息
            ranking = {}
            for metric in ["average_stability_score", "average_semantic_score", "average_position_score", "average_match_ratio"]:
                sorted_prompts = sorted(
                    [p for p in prompt_types if comparison_result["prompt_comparison"][p]["dataset_count"] > 0],
                    key=lambda p: comparison_result["prompt_comparison"][p][metric],
                    reverse=True
                )
                ranking[metric] = {
                    prompt: idx + 1 for idx, prompt in enumerate(sorted_prompts)
                }
            
            comparison_result["rankings"] = ranking
            
            # 保存比较结果
            with open("Output/Stability-Output/prompt_stability_comparison.json", "w", encoding="utf-8") as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False)
            
            # 打印比较结果
            self.logger.write_line(f"[{datetime.now()}] [INFO] prompt稳定性比较结果:")
            for prompt_type in prompt_types:
                prompt_result = comparison_result["prompt_comparison"][prompt_type]
                self.logger.write_line(
                    f"[{datetime.now()}] [INFO] {prompt_type}: "
                    f"稳定性得分={prompt_result['average_stability_score']:.2f} "
                    f"语义得分={prompt_result['average_semantic_score']:.2f} "
                    f"位置得分={prompt_result['average_position_score']:.2f} "
                    f"匹配率={prompt_result['average_match_ratio']:.2f} "
                    f"数据集数量={prompt_result['dataset_count']} "
                    f"评估对数={prompt_result['pair_count']}"
                )
            
            self.logger.write_line(f"[{datetime.now()}] [INFO] prompt稳定性排名:")
            for metric, ranks in ranking.items():
                metric_name = {
                    "average_stability_score": "稳定性得分",
                    "average_semantic_score": "语义得分",
                    "average_position_score": "位置得分",
                    "average_match_ratio": "匹配率"
                }.get(metric, metric)
                
                rank_str = ", ".join([f"{idx}:{p}" for p, idx in sorted(ranks.items(), key=lambda x: x[1])])
                self.logger.write_line(f"[{datetime.now()}] [INFO] {metric_name}排名: {rank_str}")
            
            self.logger.write_line(f"[{datetime.now()}] [INFO] prompt稳定性比较结果已保存到 Output/Stability-Output/prompt_stability_comparison.json")
            
        except Exception as e:
            self.logger.write_line(f"[{datetime.now()}] [ERROR] 比较稳定性结果时出错: {str(e)}")
            print(f"比较稳定性结果错误详情: {str(e)}")  # 打印详细错误信息帮助调试

if __name__ == "__main__":
    import asyncio
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LLM稳定性测试Pipeline")
    parser.add_argument("--prompt_types", type=str, default="all", help="要评估的prompt类型，用逗号分隔，可选值: baseline,ap,tbs,cast，默认为all(全部)")
    parser.add_argument("--compare_only", action="store_true", help="仅比较已有的评估结果，不运行测试")
    parser.add_argument("--score_only", action="store_true", help="仅评分，不运行测试")
    args = parser.parse_args()
    
    # 创建输入目录
    input_dirs = ["Input/Stability-Input/xlsx"]
    for dir_path in input_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
    pipeline = LLMStabilityPipeline()
    
    # 确定要评估的prompt类型
    prompt_types = ["baseline", "ap", "tbs", "cast"]
    if args.prompt_types != "all":
        user_types = args.prompt_types.split(",")
        prompt_types = [pt for pt in user_types if pt in ["baseline", "ap", "tbs", "cast"]]

    if args.score_only:
        pipeline.score_bullet_point_from_file("Output/Stability-Output/cast_stability_results.json")
        
    if args.compare_only:
        # 仅比较现有结果
        pipeline.compare_stability_results(None, prompt_types)
    else:
        # 运行完整pipeline
        asyncio.run(pipeline.run(prompt_types)) 