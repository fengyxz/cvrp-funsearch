import re
import logging
import time
import vrplib
import llm_model
from dotenv import load_dotenv
import os
from  evaluator import Evaluator
from prompt_config import ERROR_INFO,USER_PROMPT_ACO,USER_PROMPT,USER_PROMPT_GA, read_template_file
from prompt_generator import PromptGenerator
from w2csv import save_results_to_csv  

def read_cvrp_data(file_name, ending='.vrp'):
    if file_name.endswith(ending):
        instance = vrplib.read_instance( file_name)
        if instance:
            print(f'Successfully read {file_name}')
        else:
            print(f'Failed to read {file_name}')
    data = {}
    # Basic parameter settings
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
    
if __name__ == '__main__':
    load_dotenv()  # Automatically load the .env file in the current directory

    # class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    api_key =os.getenv("API_KEY")
    # === Step 1: Load the template for the solver ===
    CODE_TEMPLATE = read_template_file('cvrp-funsearch/cvrp/spec/LNS_template.txt')

    # === Step 2: Specify datasets to evaluate ===
    # Available options:
    # dataset_names = ['X-n110-k13','X-n115-k10','X-n120-k6','X-n139-k10']  # large instances
    dataset_names = ['A-n32-k5','A-n37-k5','A-n45-k6','A-n48-k7','A-n53-k7','A-n63-k10','A-n80-k10']  # small instances
    # dataset_names = ['A-n45-k6']  # test a single dataset

    results = []  # Store evaluation results

    # === Step 3: Initialize and run the solver for each dataset ===
    for dataset_name in dataset_names:
        model = llm_model.DsModel(api_key)  # Load the LLM backend with API key
        sample_size = 5  # Number of candidate solutions to generate
        llm = llm_model.LLM(sample_size, model)

        # Load CVRP data file based on dataset name
        data = read_cvrp_data(f'cvrp-funsearch/cvrp/data/small/{dataset_name}.vrp')
        input = DataSet(dataset_name,data)
        # Load the Prompt generator
        prompt_generator = PromptGenerator(USER_PROMPT,CODE_TEMPLATE,ERROR_INFO,"")    
        # Load the evaluator
        evaluator = Evaluator(CODE_TEMPLATE,'construction_heuristic','evaluate',input)
        # Run the LLM to get responses
        prompt = prompt_generator.generate(input.data)
        responses = llm.draw_samples(prompt)
       
        # Evaluate and save the results
        for idx, response in enumerate(responses):
            # Call the evaluator
            print("---------Answer[ from LLM ]------------")
            is_success, total_distance, routes, run_time = evaluator.analyse(response)
            
            results.append({
                "dataset": dataset_name,
                "sample_id": idx,
                "response":response,
                "is_success": is_success,
                "total_distance": total_distance if is_success else "N/A",
                "run_time": run_time,
                "routes": str(routes),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
    save_results_to_csv(results)