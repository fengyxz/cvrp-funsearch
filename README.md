# CVRP-Funsearch

Our project focuses on the Capacitated Vehicle Routing Problem (CVRP), a popular NP-hard problem. This project introduces the innovative FunSearch method to address CVRP, leveraging Large Language Models (LLM) and Retrieval-Augmented Generation (RAG) technology for external knowledge retrieval.

The approach combines deep reasoning through chain of thought to automate program generation and search within the function space. RAG ensures precise knowledge retrieval, providing a theoretical foundation and reasoning framework, while the chain of thought decomposes CVRP into interconnected subproblems for stepwise optimization, tackling both single and multi-objective challenges effectively.

## Directory Explanation

cvrp-funsearch/

```
├── LLM.ipynb
├── Other_algorithms
│   ├── 0_0_Ortools_.ipynb
│   ├── 0_1_k-opt.ipynb
│   ├── 0_2_Large_Neighborhood_Search_LNS.ipynb
│   ├── 0_3_Tabu_Search.ipynb
│   ├── 0_4_Simulated_Annealing.ipynb
│   └── gif.ipynb
├── cvrp
│   ├── code_manipulation.py
│   ├── config.py
│   ├── cvrp_funsearch.py
│   ├── evaluator.py
│   ├── evaluator_accelerate.py
│   ├── programs_database.py
│   └── sampler.py
├── data
│   ├── cvrp
│   └── tsp
├── ortools
│   ├── cvrp
│   ├── cvrp-1.ipynb
│   └── cvrp.ipynb
├── others-cvrp.ipynb
├── requirements.txt
└── test_spec.txt
```

`LLM.ipynb`: This Jupyter Notebook is likely used for experiments or demonstrations related to Large Language Models (LLM), showing how LLM can generate and optimize code for solving the CVRP problem.

`Other_algorithms/`: Contains Jupyter Notebooks that implement various other algorithms for solving CVRP, used for comparison and testing the performance of different algorithms.

`cvrp/`: Contains core CVRP-related code modules.

`data/`: Contains datasets related to the problem.

`ortools/`: Contains code and Jupyter Notebooks that use Google OR-Tools to solve CVRP.

`others-cvrp.ipynb`: A Jupyter Notebook that likely consolidates or compares the implementations and results of other CVRP solving methods.

`test_spec.txt`: Likely contains specifications or descriptions of test cases used to validate the correctness of the algorithms and modules in the project.

# Five steps

1. Implement LLM interface.

2. Implement a SandBox interface.

3. Prepare a specification.

4. Prepare a dataset.
   
5. Start FunSearch.
