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
    def __init__(self, log_path):
        self.log_path = log_path
        directory_path = os.path.dirname(log_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        self.file = open(log_path, 'a', encoding='utf-8')
    
    def write_line(self, message):
        print(message)
        self.file.write(message + '\n')
        self.file.flush()
    
    def close(self):
        self.file.close()

class Dataset:
    def __init__(self, name, language, text_items):
        self.name = name
        self.language = language
        self.text_items = text_items

class Query:
    def __init__(self, query_text, query_language, output_language=None):
        self.query_text = query_text
        self.query_language = query_language
        self.output_language = output_language or query_language

class TextStructuredInsight:
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
         
        self.siliconflow_client = OpenAI(
            api_key=os.getenv("SiliconFlow_API_KEY"), 
            base_url="https://api.siliconflow.cn/v1",
            timeout=300.0
        )
        
         
        self.grok_client = OpenAI(
            api_key=os.getenv("Grok_API_KEY"),
            base_url="https://api.x.ai/v1",
            timeout=300.0
        )

         
        self.gemini_client = genai.Client(api_key=os.getenv("Gemini_API_KEY"))

         
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=300.0
        )

         
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=300.0
        )
        
    async def text_analysis_with_nl_query(self, dataset, query, prompt_type):
        """Perform text analysis using specified LLM provider.
        
        Args:
            dataset: 数据集对象
            query: 查询对象
            prompt_type: 提示词类型
        """
        try:
            system_prompt = self.get_text_analysis_system_prompt(prompt_type)
            user_prompt = self.get_text_analysis_user_prompt(dataset.text_items, dataset.name, query.query_text, query.query_language, query.output_language)
            
             
            print(f"System prompt length: {len(system_prompt)}")
            print(f"User prompt sample: {user_prompt[:100]}...")
            llm_type = "gpt-4.1"
            response = await self.llm_requests_async(system_prompt, user_prompt, max_tokens=4096, llm_type=llm_type)
            if llm_type in ["gemini-2.5-pro-preview-03-25", "gemini-2.5-flash-preview-04-17"]:
                content = response.text
            else:
                content = response.choices[0].message.content
            
             
            print(f"API返回内容样本: {content[:200]}...")
            
             
            insight = TextStructuredInsight()
            insight.text_items = dataset.text_items
            insight.user_query = query.query_text
            insight.output_language = query.output_language
            
             
            try:
                 
                data = json.loads(content)
                
                 
                insight.general_response = content
                 
                all_possible_bullet_points = []
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        if all(isinstance(item, dict) for item in value):
                            all_possible_bullet_points = value
                            break

                if all_possible_bullet_points:
                    insight.bullet_list = all_possible_bullet_points
                
                 
                if not insight.bullet_list:
                     
                    insight.bullet_list = self._extract_bullet_points(content)
                
                 
                if not insight.general_response:
                    insight.general_response = content
                
                print(f"成功解析为JSON，bullet_points数量: {len(insight.bullet_list)}")
                
                return insight
            except json.JSONDecodeError:
                 
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    try:
                        response_json = content[json_start:json_end]
                        response_data = json.loads(response_json)
                        
                         
                        insight.general_response = response_json
                        insight.bullet_list = []
                        
                         
                        if not insight.bullet_list:
                            insight.bullet_list = self._extract_bullet_points(content)
                        
                         
                        if not insight.general_response:
                            insight.general_response = content
                        
                        print(f"成功解析为JSON（从内容中提取），bullet_points数量: {len(insight.bullet_list)}")
                        
                        return insight
                    except Exception as json_e:
                        print(f"JSON解析错误: {str(json_e)}, 内容: {response_json[:100]}...")
                
                 
                print(f"没有找到有效的JSON格式，使用原始内容")
                
                 
                insight.general_response = content
                insight.bullet_list = self._extract_bullet_points(content)
                
                print(f"从非结构化内容中提取bullet_points，数量: {len(insight.bullet_list)}")
                
                return insight
                
        except Exception as e:
            print(f"文本分析API调用错误: {str(e)}")
             
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            raise e
            
    def _extract_bullet_points(self, content):
        """从文本内容中提取可能的bullet points"""
        bullet_points = []
        
         
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or 
                         line.startswith('*') or 
                         (len(line) > 2 and line[0].isdigit() and '.' in line[:3])):
                 
                if len(line) > 2 and line[0].isdigit() and '.' in line[:3]:
                    content_part = line.split('.', 1)[1].strip()
                    bullet_points.append({
                        "Title": f"Point {len(bullet_points)+1}",
                        "Description": content_part
                    })
                else:
                     
                    content_part = line[1:].strip() if len(line) > 1 else line
                    bullet_points.append({
                        "Title": f"Point {len(bullet_points)+1}",
                        "Description": content_part
                    })
        
         
        if not bullet_points and len(lines) > 1:
            for i, line in enumerate(lines):
                if line.strip():
                    bullet_points.append({
                        "Title": f"Point {len(bullet_points)+1}",
                        "Description": line.strip()
                    })
                     
                    if len(bullet_points) >= 5:
                        break
        
         
        if not bullet_points:
            bullet_points.append({
                "Title": "Summary",
                "Description": content[:200] + ("..." if len(content) > 200 else "")
            })
        
        return bullet_points
    
    async def score_bullet_point(self, results_array):
        """评估子弹点列表的稳定性"""
        try:
             
            grouped_data = self._group_results_by_dataset_query(results_array)
            all_scores = []
            
             
            for key, group_info in list(grouped_data.items()):
                dataset_name = group_info["dataset"]
                query_text = group_info["query"]
                
                 
                all_generations = group_info["generations"]
                num_generations = len(all_generations)
                print(f"评估 '{dataset_name}' - '{query_text}'，有 {num_generations} 轮生成")
                
                 
                if num_generations < 2:
                    print(f"  跳过评估 - 生成次数不足")
                    continue
                
                 
                pair_scores = []
                 
                total_pairs = (num_generations * (num_generations - 1)) // 2
                pair_count = 0
                
                 
                for i in range(num_generations - 1):
                    for j in range(i + 1, num_generations):
                        pair_count += 1
                        print(f"  评估组合 {pair_count}/{total_pairs}: 轮次 {i+1} vs 轮次 {j+1}")
                        
                        bullet_group1 = self._convert_to_bullet_point_group(all_generations[i])
                        bullet_group2 = self._convert_to_bullet_point_group(all_generations[j])
                        
                         
                        if len(bullet_group1) == 0 or len(bullet_group2) == 0:
                            print(f"    跳过 - 某一组bullet points为空")
                            continue
                        
                         
                        system_prompt = self.get_bullet_point_stability_system_prompt()
                        
                         
                        bullet_point_data = {
                            "BulletPointGroup1": bullet_group1,
                            "BulletPointGroup2": bullet_group2
                        }
                        
                         
                        user_prompt = json.dumps(bullet_point_data, ensure_ascii=False, indent=2)
                        print(f"    发送评估请求，提示词长度: {len(user_prompt)}")
                        
                         
                        try:
                            response = await self.llm_requests_async(system_prompt, user_prompt, max_tokens=8192, llm_type="gpt-4o")
                            content = response.choices[0].message.content
                            print(f"    收到API响应，长度: {len(content)}")
                            
                             
                            json_content = ""
                            try:
                                 
                                match_result = json.loads(content)
                                json_content = content
                            except json.JSONDecodeError:
                                 
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
                            
                             
                            semantic_matches = match_result.get("SemanticMatches", [])
                            matched_positions = match_result.get("MatchedPositions", {})
                            analysis_details = match_result.get("AnalysisDetails", "")
                            
                             
                             
                            total_similarity = sum(match.get("SimilarityScore", 0) for match in semantic_matches)
                            semantic_score = total_similarity / len(semantic_matches) if semantic_matches else 0
                            
                             
                             
                            min_group_size = min(len(bullet_group1), len(bullet_group2))
                            match_ratio_min = len(semantic_matches) / min_group_size if min_group_size > 0 else 0
                            
                             
                            union_size = len(bullet_group1) + len(bullet_group2) - len(semantic_matches)
                            jaccard_index = len(semantic_matches) / union_size if union_size > 0 else 0
                            
                             
                            avg_size = (len(bullet_group1) + len(bullet_group2)) / 2
                            avg_match_ratio = len(semantic_matches) / avg_size if avg_size > 0 else 0
                            
                             
                             
                            unmatch_ratio = 1 - jaccard_index
                            
                             
                            size_difference = abs(len(bullet_group1) - len(bullet_group2)) / max(len(bullet_group1), len(bullet_group2)) if max(len(bullet_group1), len(bullet_group2)) > 0 else 0
                            
                             
                             
                            penalty_score = (unmatch_ratio * 0.8 + size_difference * 0.2)
                             
                             
                            
                            print(f"    计算得到 SemanticScore: {semantic_score:.2f}, Jaccard指数: {jaccard_index:.2f}, 未匹配惩罚: {penalty_score:.2f}")
                            print(f"    原始匹配率: {match_ratio_min:.2f}, 平均匹配率: {avg_match_ratio:.2f}, 大小差异: {size_difference:.2f}")
                            
                             
                            kendall_tau, p_value, position_consistency = self._calculate_kendall_tau_from_matched_positions(matched_positions)
                            print(f"    计算Kendall tau相关系数: {kendall_tau:.4f}, p值: {p_value:.4f}, 位置一致性: {position_consistency:.4f}")
                            
                             
                             
                            semantic_weight = 0.6
                            position_weight = 0.4
                            
                             
                            normalized_semantic_score = semantic_score / 5
                            
                             
                            content_score = (normalized_semantic_score * semantic_weight) + (position_consistency * position_weight)
                            
                             
                             
                             
                             
                            max_penalty_effect = 0.5   
                            penalty_factor = 1 - (penalty_score * max_penalty_effect)
                            
                             
                            total_score = content_score * penalty_factor
                            
                             
                            total_score = max(0, min(1, total_score))
                            
                             
                            stability_score = total_score * 10
                            
                             
                            semantic_score_10 = normalized_semantic_score * 10
                            position_score_10 = position_consistency * 10
                            jaccard_index_10 = jaccard_index * 10
                            penalty_score_10 = penalty_score * 10
                            
                             
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
                
                 
                if pair_scores:
                     
                    avg_stability = sum(item["stability_score"] for item in pair_scores) / len(pair_scores)
                    avg_semantic = sum(item["semantic_score"] for item in pair_scores) / len(pair_scores)
                    avg_position = sum(item["position_score"] for item in pair_scores) / len(pair_scores)
                    avg_match_ratio = sum(item["average_match_ratio"] for item in pair_scores) / len(pair_scores)
                    
                     
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
            
             
            if all_scores:
                with open("Output/Stability-Output/bullet_point_stability_score.json", "w", encoding="utf-8") as f:
                    json.dump(all_scores, f, indent=2, ensure_ascii=False)
                
                 
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
            
    async def llm_requests_async(self, system_prompt, user_prompt, max_tokens=4096, llm_type=None, seed=42):
        """LLM API请求"""
        try:
            if llm_type in ["gpt-4o", "o4-mini", "gpt-4.1"]:
                response = self.openai_client.chat.completions.create(
                    model=llm_type,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                     
                     
                     
                     
                     
                )
            elif llm_type in ["grok-3-beta"]:
                response = self.grok_client.chat.completions.create(
                    model=llm_type,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}    
                    ],
                    max_tokens=max_tokens,
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
                     
                     
                     
                     
                     
                )
            else:
                raise ValueError(f"不支持的LLM类型: {llm_type}")
            
            return response
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            raise e
    
    def get_text_analysis_system_prompt(self, prompt_type):
        """获取文本分析系统提示词
        
        Args:
            prompt_type: 提示词类型
        """
        try:
             
            reasoning_path = os.path.join("reasoning_path_prompt", f"{prompt_type}.md")
            if not os.path.exists(reasoning_path):
                raise FileNotFoundError(f"找不到提示词文件: {reasoning_path}")
                
            print(f"正在加载reasoning path提示词文件: {reasoning_path}")
            with open(reasoning_path, "r", encoding="utf-8") as f:
                prompt = f.read()
            return prompt
            
        except Exception as e:
            print(f"读取文本分析提示词文件出错: {str(e)}")
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
         
        if not results_array or len(results_array) == 0:
            print("警告: results_array为空，无法生成稳定性评估提示")
            return json.dumps({"BulletPointGroup1": [], "BulletPointGroup2": []})
            
        print(f"处理results_array，包含{len(results_array)}组结果")
        
         
        grouped_data = self._group_results_by_dataset_query(results_array)
        print(f"分组后有{len(grouped_data)}个数据集-查询对")
        
         
        for key, group_info in list(grouped_data.items()):
             
            all_generations = group_info["generations"]
            print(f"数据集-查询对 '{key}' 有 {len(all_generations)} 轮生成")
            
             
            if len(all_generations) >= 2:
                 
                bullet_group1 = self._convert_to_bullet_point_group(all_generations[0])
                bullet_group2 = self._convert_to_bullet_point_group(all_generations[1])
                
                print(f"第一组bullet points: {len(bullet_group1)}个")
                print(f"第二组bullet points: {len(bullet_group2)}个")
                
                 
                if len(bullet_group1) == 0 or len(bullet_group2) == 0:
                    print(f"警告: 数据集-查询对 '{key}' 的bullet points为空，跳过")
                    continue
                
                 
                bullet_point_data = {
                    "BulletPointGroup1": bullet_group1,
                    "BulletPointGroup2": bullet_group2
                }
                
                 
                user_prompt = json.dumps(bullet_point_data, ensure_ascii=False, indent=2)
                print(f"创建user_prompt (长度: {len(user_prompt)})")
                return user_prompt
        
         
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
                 
                print(f"处理第{i+1}个bullet point，类型: {type(point)}")
                
                 
                if isinstance(point, dict):
                    bullet_point = point.copy()
                    
                     
                    if "Title" not in bullet_point and "title" in bullet_point:
                        bullet_point["Title"] = bullet_point["title"]
                    if "Description" not in bullet_point and "description" in bullet_point:
                        bullet_point["Description"] = bullet_point["description"]
                    
                     
                    if "Title" not in bullet_point:
                        for k, v in bullet_point.items():
                            if isinstance(v, str) and k != "Description":
                                bullet_point["Title"] = v
                                break
                    
                     
                    if "Title" not in bullet_point:
                        bullet_point["Title"] = list(bullet_point.keys())[0] if bullet_point else "Point"
                    
                     
                    if "Position" not in bullet_point:
                        bullet_point["Position"] = i
                    
                    formatted_bullet_points.append(bullet_point)
                elif isinstance(point, str):
                     
                    title_desc = self._extract_title_description(point)
                    formatted_bullet_points.append({
                        "Title": title_desc["title"],
                        "Description": title_desc["description"],
                        "Position": i
                    })
                else:
                     
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
         
        separators = [": ", " - ", ". ", "、", "，", "。"]
        
        for sep in separators:
            if sep in bullet_point_text:
                parts = bullet_point_text.split(sep, 1)
                return {"title": parts[0].strip(), "description": parts[1].strip()}
        
         
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
             
            group1_positions = matched_positions.get("Group1Positions", [])
            group2_positions = matched_positions.get("Group2Positions", [])
            
             
            if len(group1_positions) != len(group2_positions) or len(group1_positions) < 2:
                print(f"匹配位置数据不完整或不足: G1={group1_positions}, G2={group2_positions}")
                return 0, 1.0, 0
                
             
            tau, p_value = stats.kendalltau(group1_positions, group2_positions)
            
             
            if math.isnan(tau):
                return 0, 1.0, 0
                
             
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
        
         
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(base_dir, "Input/Stability-Input/xlsx/Multilingual Text Summarization Datasets.xlsx")
        self.query_path = os.path.join(base_dir, "Input/Stability-Input/xlsx/Multilingual NL Queries without Column Name.xlsx")
        
    async def run(self, prompt_types):
        """运行稳定性测试管道"""
        self.logger.write_line(f"[{datetime.now()}] [INFO] Starting LLM Stability Pipeline")
        
         
        stability_result_path = "Output/Stability-Output/stability_results.json"
        directory_path = os.path.dirname(stability_result_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        
         
        await self.score_bullet_point(stability_result_path, prompt_types)
        
        self.logger.write_line(f"[{datetime.now()}] [INFO] LLM Stability Pipeline completed")
        self.logger.close()
        
    async def score_bullet_point(self, score_bullet_point_result, prompt_types):
        """评估子弹点生成的稳定性"""
         
        datasets = self.get_datasets_from_excel(self.dataset_path)
        queries = self.get_queries_from_excel(self.query_path)
        
         
        round_times = 1   
        
         
        instances = []
        for dataset in datasets:
            for query in queries:
                if dataset.language == query.query_language:
                    instances.append({"Dataset": dataset, "Query": query})
        
        self.logger.write_line(f"[{datetime.now()}] [INFO] Dataset Number: {len(instances)}")
        
        all_results = {}
        
         
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
                            
                             
                             
                                #json.dump(final_result, f, indent=2, ensure_ascii=False)
                                
                            break
                        except Exception as e:
                            self.logger.write_line(
                                f"[{datetime.now()}] [ERROR] PromptType: {prompt_type}, "
                                f"Dataset: {instance['Dataset'].name}, "
                                f"Language: {instance['Dataset'].language}, "
                                f"Query: {instance['Query'].query_text}, "
                                f"Error: {str(e)}"
                            )
                            
                             
                            if "Throttling" in str(e):
                                time.sleep(60)
                            else:
                                time.sleep(5)
            
             
             
             
            
             
             
             
                
             
            
         
         
    
    def score_bullet_point_from_file(self, file_path):
        """从文件中读取结果并评分"""
        import asyncio
        
        if not os.path.exists(file_path):
            self.logger.write_line(f"[{datetime.now()}] [ERROR] File not found: {file_path}")
            return
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                final_result = json.load(f)
                
             
            stability_score_result = asyncio.run(self.llm_api.score_bullet_point(final_result))
            
             
            output_path = "Output/Stability-Output/stability_score_from_file.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(stability_score_result, f, indent=2, ensure_ascii=False)
                
             
            simplified_scores = []
            for score in stability_score_result:
                simplified_score = {k: v for k, v in score.items() if k != "pair_details"}
                simplified_scores.append(simplified_score)
                
            with open("Output/Stability-Output/stability_score_from_file_summary.json", "w", encoding="utf-8") as f:
                json.dump(simplified_scores, f, indent=2, ensure_ascii=False)
                
             
            if stability_score_result:
                 
                avg_stability = sum(item.get("stability_score", 0) for item in stability_score_result) / len(stability_score_result)
                avg_semantic = sum(item.get("semantic_score", 0) for item in stability_score_result) / len(stability_score_result)
                avg_position = sum(item.get("position_score", 0) for item in stability_score_result) / len(stability_score_result)
                avg_jaccard = sum(item.get("jaccard_index", 0) for item in stability_score_result) / len(stability_score_result)
                avg_penalty = sum(item.get("penalty_score", 0) for item in stability_score_result) / len(stability_score_result)
                
                 
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
            print(f"错误详情: {str(e)}")   
    
    def get_datasets_from_excel(self, path):
        """从Excel文件中读取数据集"""
         
        language_map = {
            "english": "en_US",
            "spanish": "es_ES",
            "french": "fr_FR",
            "german": "de_DE",
            "italian": "it_IT",
            "japanese": "ja_JP",
            "chinese": "zh_CN",
             
        }
        
        datasets = []
        try:
             
            excel_file = pd.ExcelFile(path)
            
            for sheet_name in excel_file.sheet_names:
                 
                parts = sheet_name.split('_')
                if len(parts) >= 2:
                    name = parts[0]
                    language_key = parts[1].lower()
                    language = language_map.get(language_key, "en_US")
                    
                     
                    df = excel_file.parse(sheet_name)
                    
                     
                    last_column = df.columns[-1]
                    text_items = df[last_column].dropna().tolist()
                    
                     
                    if len(text_items) > 0:
                        datasets.append(Dataset(name, language, text_items))
            
            return datasets
        except Exception as e:
            print(f"读取数据集时出错: {str(e)}")
            return []
    
    def get_datasets_from_benchmark(self, benchmark_name):
        """从基准测试中读取数据集"""
        if benchmark_name == "Yelp":
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
         
        language_map = {
            "english": "en_US",
            "spanish": "es_ES",
            "french": "fr_FR",
            "german": "de_DE",
            "italian": "it_IT",
            "japanese": "ja_JP",
            "chinese": "zh_CN",
             
        }
        
        queries = []
        try:
             
            excel_file = pd.ExcelFile(path)
            
            for sheet_name in excel_file.sheet_names:
                 
                language = language_map.get(sheet_name.lower(), "en_US")
                
                 
                df = excel_file.parse(sheet_name)
                
                 
                for _, row in df.iloc[1:].iterrows():
                     
                    columns_to_check = [0, 1, 4]   
                    
                    for col_idx in columns_to_check:
                        if col_idx < len(df.columns):
                            query_text = row.iloc[col_idx]
                            if isinstance(query_text, str) and query_text.strip():
                                 
                                match = re.search(r'<(.+?)>', query_text)
                                
                                if match:
                                     
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
            
             
            stability_scores = {}
            for prompt_type in prompt_types:
                try:
                    with open(f"Output/Stability-Output/{prompt_type}_stability_score_result.json", "r", encoding="utf-8") as f:
                        stability_scores[prompt_type] = json.load(f)
                except Exception as e:
                    self.logger.write_line(f"[{datetime.now()}] [ERROR] 无法加载{prompt_type}的稳定性评分结果: {str(e)}")
                    stability_scores[prompt_type] = []
            
             
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
            
             
            with open("Output/Stability-Output/prompt_stability_comparison.json", "w", encoding="utf-8") as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False)
            
             
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
            print(f"比较稳定性结果错误详情: {str(e)}")   

if __name__ == "__main__":
    import asyncio
    import argparse
    
     
    parser = argparse.ArgumentParser(description="LLM稳定性测试Pipeline")
    parser.add_argument("--prompt_types", type=str, default="all", help="要评估的prompt类型，用逗号分隔，可选值: perspective_prompt,num_of_text_items_prompt,num_of_bullet_points_prompt,domain_prompt,full_cast_prompt,minimal_prompt,topwords_only_prompt,structured_reasoning_prompt，默认为all(全部)")
    parser.add_argument("--compare_only", action="store_true", help="仅比较已有的评估结果，不运行测试")
    parser.add_argument("--extended_cast", action="store_true", help="运行扩展的CAST验证实验")
    args = parser.parse_args()
    
     
    input_dirs = ["Input/Stability-Input/xlsx"]
    for dir_path in input_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    pipeline = LLMStabilityPipeline()
    
     
    original_prompts = ["perspective_prompt", "num_of_text_items_prompt", "num_of_bullet_points_prompt", "domain_prompt"]
    extended_prompts = ["full_cast_prompt", "minimal_prompt", "topwords_only_prompt", "structured_reasoning_prompt"]
    all_prompt_types = original_prompts + extended_prompts
    
    if args.extended_cast:
         
        prompt_types = all_prompt_types
        print("🚀 运行扩展的CAST验证实验，包含以下prompt类型:")
        for i, pt in enumerate(prompt_types, 1):
            print(f"  {i}. {pt}")
    elif args.prompt_types != "all":
        user_types = args.prompt_types.split(",")
        prompt_types = [pt for pt in user_types if pt in all_prompt_types]
        print(f"📝 运行指定的prompt类型: {prompt_types}")
    else:
        prompt_types = original_prompts
        print(f"📝 运行原有的prompt类型: {prompt_types}")
        
    if args.compare_only:
         
        print("📊 仅比较现有结果，不运行新测试")
        pipeline.compare_stability_results(None, prompt_types)
    else:
         
        print("🔬 运行完整稳定性测试pipeline")
        asyncio.run(pipeline.run(prompt_types)) 