
from multiprocessing import process
import re
import textwrap
import logging

import vrplib
import sand_box
import llm_model
from dotenv import load_dotenv
import os
from  evaluator import Evaluator
from prompt_config import CODE_TEMPLATE, ERROR_INFO,USER_PROMPT
from prompt_generator import PromptGenerator

def read_cvrp_data(file_name, ending='.vrp'):
    if file_name.endswith(ending):
        instance = vrplib.read_instance( file_name)
        if instance:
            print(f'Successfully read {file_name}')
        else:
            print(f'Failed to read {file_name}')
    data = {}
    # 基础参数设置
    data["vehicle_capacity"] = instance['capacity']
    data["num_vehicles"] = int(re.search(r'k(\d+)', instance['name']).group(1))
    data["depot"] = 0
    data['locations'] = [tuple(row) for row in instance['node_coord'].tolist()]
    data["num_locations"] = len(data["locations"])
    data['demand'] = instance['demand']
    data['distance_matrix']= instance['edge_weight']
    return data

class DataSet:
    def __init__(self, name, data):
        self.name = name
        self.data = data
    
# ----------------TEST------------------
if __name__ == '__main__':
    load_dotenv()  # 自动加载当前目录的 .env 文件

    # class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    api_key =os.getenv("API_KEY")
    
    # 加载LLM-model
    model = llm_model.DsModel(api_key)
    # 设置采样次数
    sample_size = 1
    llm = llm_model.LLM(sample_size,model)
    # 读入数据
    data = read_cvrp_data('cvrp-funsearch/cvrp/data/cvrp/small/A-n32-k5.vrp')
    input = DataSet('A-n32-k5',data)
    # 加载Prompt生成器
    prompt_generator = PromptGenerator(USER_PROMPT,CODE_TEMPLATE,ERROR_INFO,"")    
    # 加载evaluator
    evaluator = Evaluator(CODE_TEMPLATE,'construction_heuristic','evaluate',input)
    # 运行 LLM获得回答
    prompt = prompt_generator.generate(input.data)
    response = llm._draw_sample(prompt)
    # 调用evaluator
    print(response)
    answer = evaluator.analyse(response)
    