 
 

import os
import json
import time
import pandas as pd
from datetime import datetime
from openai import OpenAI
from google import genai
import asyncio
 

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
        self.text_items = []
        self.user_query = ""
        self.output_language = ""
        self.domain = ""   

class LLMAPI:
    """Unified API client for multiple LLM providers."""
    def __init__(self, prompt_file="tagging_prompt.md"):
         
        self.grok_api_key = os.getenv("GROK_API_KEY", "")
        self.siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        
         
        self.grok_client = OpenAI(
            api_key=self.grok_api_key,
            base_url="https://api.x.ai/v1",
        )
        self.siliconflow_client = OpenAI(
            api_key=self.siliconflow_api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        self.prompt_file = prompt_file
        self.openai_client = OpenAI(api_key=self.openai_api_key,base_url="https://openrouter.ai/api/v1")

    async def text_analysis_with_nl_query(self, text_items, dataset_name, query_text, query_language, llm_type="grok-3-beta"):
        """Perform tagging analysis using specified LLM provider."""
        try:
             
            system_prompt = self.get_tagging_system_prompt()
            user_prompt = self.get_tagging_user_prompt(text_items, dataset_name, query_text, query_language, query_language)
            
             
            if llm_type == "grok-3-beta":
                response = await self.llm_requests_async(system_prompt, user_prompt, max_tokens=16000, llm_type=llm_type)
            elif llm_type in ["deepseek-ai/DeepSeek-V3","deepseek-ai/DeepSeek-R1", "Qwen/Qwen2.5-72B-Instruct","Qwen/Qwen3-235B-A22B"]:
                response = await self.siliconflow_llm_requests_async(system_prompt, user_prompt, llm_type=llm_type)
            elif llm_type in ["gpt-4.1","o4-mini"]:
                response = await self.openai_llm_requests_async(system_prompt, user_prompt, llm_type=llm_type)
            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}")
            
             
            content = response.choices[0].message.content if llm_type not in ["gemini-2.5-flash-preview-04-17","gemini-2.5-pro-preview-05-06"] else response.text
            return content
            try:
                data = json.loads(content)
                insight = TextStructuredInsight()
                insight.general_response = content
                 
                 
                 
                 
                 
                return insight
            except json.JSONDecodeError:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    try:
                        response_json = content[json_start:json_end]
                        response_data = json.loads(response_json)
                        insight = TextStructuredInsight()
                        insight.general_response = response_data.get("summary", "")
                        insight.bullet_list = response_data.get("tags", [])
                        insight.text_items = text_items
                        insight.user_query = query_text
                        insight.output_language = query_language
                        insight.domain = response_data.get("domain", "")
                        return insight
                    except Exception as json_e:
                        print(f"JSON parsing error: {str(json_e)}, content: {response_json[:100]}...")
                
                print("No valid JSON found, using raw content")
                insight = TextStructuredInsight()
                tags = []
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('•') or line.startswith('-') or line.startswith('*') or (line[0].isdigit() and '.' in line[:3])):
                        tags.append(line)
                insight.general_response = content
                insight.bullet_list = tags
                insight.text_items = text_items
                insight.user_query = query_text
                insight.output_language = query_language
                insight.domain = ""
                return insight

        except Exception as e:
            print(f"Tagging analysis API error: {str(e)}")
            raise e

    async def llm_requests_async(self, system_prompt, user_prompt, max_tokens=16000, llm_type="grok-3-beta", seed=42):
        """Send request to Grok API."""
        try:
            if llm_type != "grok-3-beta":
                raise ValueError(f"Only grok-3-beta supported, received: {llm_type}")
            
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
                seed=42
            )
            return response
        except Exception as e:
            print(f"API call error: {str(e)}")
            raise e
    async def openai_llm_requests_async(self, system_prompt, user_prompt, llm_type="gpt-4.1"):
        """Send request to OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model=llm_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=16000
            )
            return response
        except Exception as e:
            print(f"OpenAI API call error: {str(e)}")
            raise e
    async def siliconflow_llm_requests_async(self, system_prompt, user_prompt, llm_type="Qwen/Qwen2.5-72B-Instruct"):
        """Send request to SiliconFlow API."""
        response = self.siliconflow_client.chat.completions.create(
            model=llm_type,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False
        )
        return response

     
         
         
         
         

         
         
         
         
                
         
         
         
         
         
         
         
         
         
         
         
         
    def get_tagging_system_prompt(self):
        """Load system prompt from markdown file."""
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read()
            return prompt
        except Exception as e:
            print(f"Error reading tagging prompt: {str(e)}")
            return "You are a tagging assistant. Generate tags for the provided text items based on the user query. Return JSON with 'summary', 'tags', and optional 'domain'."

    def get_tagging_user_prompt(self, text_items, column_name, query_text, query_language, output_language):
        """Generate user prompt with text items and query details."""
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

class TaggingPipeline:
    """Main pipeline for executing tagging experiments."""
    def __init__(self, log_path, prompt_file="tagging_prompt.md"):
        directory_path = os.path.dirname(log_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        self.logger = Logger(log_path)
        self.llm_api = LLMAPI(prompt_file)
        self.dataset_path = "Output/Stability/CombinedDataset.xlsx"

    async def go_tagging(self, save_path, sheet_name, repeat_times=5, llm_type="grok-3-beta", timing_stats=None, prompt_type=None):
        """Execute tagging pipeline for specified dataset and queries."""
        self.logger.write_line(f"[{datetime.now()}] [INFO] Starting Tagging Pipeline for {sheet_name} with {llm_type}")

        datasets = self.get_datasets_from_excel(self.dataset_path, sheet_name)
        queries = self.get_queries_from_excel(self.dataset_path, sheet_name)
        instances = [(d, q) for d in datasets for q in queries if d.language == q.query_language]

        self.logger.write_line(f"[{datetime.now()}] [INFO] Number of query and dataset pairs: {len(instances)}")

        query_index = 0
        for instance in instances:
            dataset, query = instance
            await self.gogo_tagging(save_path, dataset, query, repeat_times, query_index, llm_type, timing_stats, prompt_type)
            query_index += 1

        self.logger.write_line(f"[{datetime.now()}] [INFO] Tagging Pipeline for {sheet_name} completed")
   
    async def gogo_tagging(self, save_path, dataset, query, repeat_times, query_index, llm_type="grok-3-beta", timing_stats=None, prompt_type=None):
        """Process a single dataset-query pair with multiple repetitions."""
        results = []
        for i in range(repeat_times):
            try:
                await asyncio.sleep(5)
                start_time = datetime.now()
                analysis_result = await self.llm_api.text_analysis_with_nl_query(
                    dataset.text_items, dataset.name, query.query_text, query.query_language, llm_type
                )
                end_time = datetime.now()
                analysis_time = (end_time - start_time).total_seconds()
                await asyncio.sleep(5)

                if not analysis_result:
                    raise Exception("Analysis result is empty.")

                results.append(analysis_result)

                 
                if timing_stats is not None and prompt_type is not None:
                    timing_stats[prompt_type].append(analysis_time)

                self.logger.write_line(
                    f"[{datetime.now()}] [INFO] Dataset: {dataset.name}, "
                    f"Language: {dataset.language}, Query: {query.query_text}, "
                    f"Time: {analysis_time}s, Repeat: {i + 1}, LLM: {llm_type}"
                )
            except Exception as e:
                self.logger.write_line(
                    f"[{datetime.now()}] [ERROR] Dataset: {dataset.name}, "
                    f"Language: {dataset.language}, Query: {query.query_text}, "
                    f"Error: {str(e)}, LLM: {llm_type}"
                )
                if "Throttling" in str(e):
                    await asyncio.sleep(60)
                else:
                    await asyncio.sleep(5)

        save_file = f"{save_path}/{dataset.name}_{query_index}.json"
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.write_line(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Analysis Results saved to '{os.path.abspath(save_file)}'"
        )

    def get_datasets_from_excel(self, path, sheet_name):
        """Extract datasets from Excel file."""
        datasets = []
        try:
            excel_file = pd.ExcelFile(path)
            if sheet_name not in excel_file.sheet_names:
                raise ValueError(f"Sheet '{sheet_name}' not found in {path}")
            
            df = excel_file.parse(sheet_name)
            dataset = Dataset(
                name=sheet_name,
                language="en_US",
                text_items=[]
            )
            
             
            for text_item in df.iloc[:, 0]:
                if pd.notna(text_item) and str(text_item).strip():
                    dataset.text_items.append(str(text_item))
            
            if dataset.text_items:
                datasets.append(dataset)
            return datasets
        except Exception as e:
            print(f"Error reading datasets: {str(e)}")
            return []

    def get_queries_from_excel(self, path, sheet_name):
        """Extract queries from Excel file."""
        queries = []
        try:
            excel_file = pd.ExcelFile(path)
            if sheet_name not in excel_file.sheet_names:
                raise ValueError(f"Sheet '{sheet_name}' not found in {path}")
            
            df = excel_file.parse(sheet_name)
             
            for col in range(1, len(df.columns), 2):
                query_text = df.iloc[0, col]
                if pd.notna(query_text) and str(query_text).strip():
                    queries.append(Query(
                        query_text=str(query_text),
                        query_language="en_US"
                    ))
            return queries
        except Exception as e:
            print(f"Error reading queries: {str(e)}")
            return []

async def main():
    """Main execution function for tagging experiments."""
    dataset_path = "Output/Stability/CombinedDataset.xlsx"
    sheet_names = ["Amazon_100_2"]
    prompt_files = ["AP+TbS.md","none.md"]
    llm_types = ["o4-mini"]
    
     
    timing_stats = {prompt_file.split('.')[0]: [] for prompt_file in prompt_files}
    
     
    for llm_type in llm_types:
        for sheet_name in sheet_names:
            for prompt_file in prompt_files:
                prompt_type = prompt_file.split('.')[0]
                save_path = f"try/{llm_type.replace('/', '_')}/{sheet_name}/{prompt_type}/"
                log_path = f"try/{llm_type.replace('/', '_')}/{sheet_name}/{prompt_type}/log.txt"

                pipeline = TaggingPipeline(log_path, prompt_file)
                await pipeline.go_tagging(save_path, sheet_name, 2, llm_type, timing_stats, prompt_type)
                pipeline.logger.close()

     
    print("\n=== API Response Time Statistics (seconds) ===")
    for prompt, times in timing_stats.items():
        print(f"{prompt}: " + " ".join(f"{t:.2f}" for t in times))

if __name__ == "__main__":
    asyncio.run(main())