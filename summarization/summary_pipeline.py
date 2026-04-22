import os
import json
import time
import re
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from openai import OpenAI

 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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
        self.output_language = output_language

class NewScoreOutput:
    """Stores evaluation scores for summarization quality."""
    def __init__(self):
        self.precision = 0.0
        self.recall = 0.0
        self.score_reason = ""
        self.content_score = 0.0
        self.clear_boundary_score = 0.0
        self.balance_score = 0.0
        self.coverage_score = 0.0
        self.config_score = 0.0
        self.is_hallucinated = False
        self.config = ""
        self.filter = ""
        self.rate_summary = ""
        self.detected_language = ""

class LLMAPI:
    """API client for LLM interactions."""
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=300.0
        )
        
    def initiate_token(self):
        """Initialize API token (not needed in Python version)."""
        pass
        
    async def llm_requests_async(self, system_prompt, user_prompt, max_tokens=4096, llm_type="openai/gpt-4.1", seed=42):
        """Send request to LLM API."""
        try:
            response = self.client.chat.completions.create(
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
            return response
        except Exception as e:
            print(f"API call error: {str(e)}")
            raise e
            
    async def score_text_analysis_new(self, dataset, query, analysis_result):
        """Evaluate summarization quality using LLM-as-a-Judge."""
        prompt = self.get_new_score_text_analysis_prompt()
        prompt = prompt.replace("{textItems}", self.text_items_to_string(dataset.text_items))
        prompt = prompt.replace("{userQuery}", query.query_text)
        prompt = prompt.replace("{analysisResult}", analysis_result)
        
        result = await self.llm_requests_async(prompt, "", llm_type="openai/gpt-4.1")
        
        try:
            score_json = result.choices[0].message.content
             
            score_json = re.search(r'\{.*\}', score_json, re.DOTALL).group()
            score_output = json.loads(score_json)
            
            output = NewScoreOutput()
            output.precision = score_output.get("Precision", 0.0)
            output.recall = score_output.get("Recall", 0.0)
            output.score_reason = score_output.get("Score Reason", "")
            output.content_score = score_output.get("Content Score", 0.0)
            output.clear_boundary_score = score_output.get("Clear Boundary Score", 0.0)
            output.balance_score = score_output.get("Balance Score", 0.0)
            output.coverage_score = score_output.get("Coverage Score", 0.0)
            output.config_score = score_output.get("Config Score", 0.0)
            output.is_hallucinated = output.precision != 1.0
            output.config = score_output.get("Config", "")
            output.filter = score_output.get("Filter", "")
            output.rate_summary = score_output.get("Summary", "")
            
             
            output.detected_language = "en"  
            
            return output
        except Exception as e:
            print(f"解析评分输出失败: {str(e)}")
            raise Exception(f"Failed to parse the JSON: {result}")
    
    def get_new_score_text_analysis_prompt(self):
        with open("EvaluationPrompt/summarization_evaluation_prompt.md", "r", encoding="utf-8") as file:
            new_score_text_analysis_prompt = file.read()
        return new_score_text_analysis_prompt
    
    def text_items_to_string(self, text_items):
        """将文本项转换为格式化字符串"""
        result = []
        for i, item in enumerate(text_items):
            result.append(f"[{i+1}] {item}")
        return "\n".join(result)
    
    def get_summarization_system_prompt(self, text_items, column_name, query_text, query_language):
        """生成摘要的系统提示词"""
        with open("AblationPrompt/baseline_prompt.md", "r", encoding="utf-8") as file:
            summarization_system_prompt = file.read()
        return summarization_system_prompt
    
    def get_summarization_user_prompt(self, text_items, column_name, query_text, query_language):
        """生成摘要的用户提示词"""
        prompt = f"User Query: {query_text}\n"
        prompt += f"Query Language: {query_language}\n"
        if column_name:
            prompt += f"Column Name: {column_name}\n"
        prompt += "Text Items:\n"
        
        for i, item in enumerate(text_items):
            prompt += f"[{i+1}] {item}\n"
            
        return prompt

class ChatBasedSummaryPipeline:
    def __init__(self, log_path="Output/Summary-Output/log.txt"):
        directory_path = os.path.dirname(log_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
        self.logger = Logger(log_path)
        self.llm_api = LLMAPI()
        self.llm_api.initiate_token()
        
        self.dataset_path = "Input/Stability-Input/xlsx/Multilingual Text Summarization Datasets.xlsx"
        self.query_path = "Input/Stability-Input/xlsx/Multilingual NL Queries without Column Name.xlsx"
    
    def run_summary(self, output_path="Output/Summary-Output/results.json"):
        directory_path = os.path.dirname(output_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
        datasets = self.get_datasets_from_excel(self.dataset_path)
        queries = self.get_queries_from_excel(self.query_path)
        
        import asyncio
        asyncio.run(self.evaluate_summary(output_path, datasets, queries))
    
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
    
    async def evaluate_summary(self, output_path, datasets, queries):
        results = []
        
         
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except:
                results = []
        
         
        instances = []
        for dataset in datasets:
            for query in queries:
                 
                is_processed = any(
                    r.get("Dataset") == f"{dataset.name}_{dataset.language}" and 
                    r.get("Query") == query.query_text
                    for r in results
                )
                
                if not is_processed:
                    instances.append({"Dataset": dataset, "Query": query})
        
        self.logger.write_line(f"[{datetime.now()}] [INFO] Number of query and dataset pairs: {len(instances)}")
        
        for instance in instances:
            while True:
                try:
                    start_time = datetime.now()
                    
                     
                    system_prompt = self.llm_api.get_summarization_system_prompt(
                        instance["Dataset"].text_items, 
                        instance["Dataset"].name,
                        instance["Query"].query_text,
                        instance["Query"].query_language
                    )
                    
                    user_prompt = self.llm_api.get_summarization_user_prompt(
                        instance["Dataset"].text_items,
                        instance["Dataset"].name,
                        instance["Query"].query_text,
                        instance["Query"].query_language
                    )
                    
                     
                    response = await self.llm_api.llm_requests_async(
                        system_prompt, 
                        user_prompt,
                        max_tokens=4096,
                        llm_type="openai/gpt-4.1",
                        seed=42
                    )
                    
                    raw_summary_result = response.choices[0].message.content
                    
                     
                    json_start = raw_summary_result.find('{')
                    json_end = raw_summary_result.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        raw_summary_result = raw_summary_result[json_start:json_end]
                        summary_result = [json.loads(raw_summary_result)]
                    else:
                         
                        summary_result = [{"summary": raw_summary_result}]
                    
                    end_time = datetime.now()
                    summary_time = (end_time - start_time).total_seconds()
                    
                     
                    time.sleep(5)
                    
                     
                    general_response = json.dumps(summary_result)
                    general_response = re.sub(r'\[\d+\]', '', general_response)
                    
                     
                    score_output = await self.llm_api.score_text_analysis_new(
                        instance["Dataset"],
                        instance["Query"],
                        general_response
                    )
                    
                     
                    result = {
                        "Dataset": f"{instance['Dataset'].name}_{instance['Dataset'].language}",
                        "Query": instance["Query"].query_text,
                        "QueryLanguage": instance["Query"].query_language,
                        "OutputLanguage": instance["Query"].output_language or instance["Query"].query_language,
                        "SLMOutputLanguage": score_output.detected_language,
                        "AnalysisResult": general_response,
                        "ScoreResult": {
                            "Precision": score_output.precision,
                            "Recall": score_output.recall,
                            "Score Reason": score_output.score_reason,
                            "Content Score": score_output.content_score,
                            "Clear Boundary Score": score_output.clear_boundary_score,
                            "Balance Score": score_output.balance_score,
                            "Coverage Score": score_output.coverage_score,
                            "Config Score": score_output.config_score,
                            "Filter": score_output.filter,
                            "Summary": score_output.rate_summary
                        },
                        "Time": summary_time
                    }
                    
                    results.append(result)
                    
                    self.logger.write_line(
                        f"[{datetime.now()}] [INFO] Dataset: {instance['Dataset'].name}, "
                        f"Language: {instance['Dataset'].language}, "
                        f"Query: {instance['Query'].query_text}, "
                        f"Time: {summary_time}s"
                    )
                    
                     
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    
                     
                    time.sleep(15)
                    
                     
                    break
                    
                except Exception as e:
                    self.logger.write_line(
                        f"[{datetime.now()}] [ERROR] Dataset: {instance['Dataset'].name}, "
                        f"Language: {instance['Dataset'].language}, "
                        f"Query: {instance['Query'].query_text}, "
                        f"Error: {str(e)}"
                    )
                    
                     
                    if "Throttling" in str(e):
                        time.sleep(60)
                    else:
                        time.sleep(5)
        
        self.logger.write_line(
            f"[{datetime.now()}] [INFO] Results are saved to {os.path.abspath(output_path)}"
        )

if __name__ == "__main__":
    pipeline = ChatBasedSummaryPipeline()
    pipeline.run_summary() 