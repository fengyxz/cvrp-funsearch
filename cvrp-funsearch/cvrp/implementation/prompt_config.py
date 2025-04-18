  
# 用户指令
# USER_PROMPT = """
# Find the better CVRP solution, try to change @funsearch.evolve:  construction_heuristic
# just only change and reture tihis function's mame
# only return the implementation of  @funsearch.evolve: construction_heuristic
# please sure your is correct python code and just provide function `construction_heuristic` do not include any examples or extraneous functions.

# Here is the code template, dataset as follows\n
# IMPORTANT: Your response should only involve the function  `construction_heuristic`, don't give other code.
# ATTENTION:@dataclass
# class ConstructionContext:
#     depot: int
#     candidate: int
#     distance: float
#     demand: int
#     vehicle_load: int
#     vehicle_capacity: int
#     locations: np.ndarray
# """
USER_PROMPT_ACO = """
Find the better CVRP solution, try to change @funsearch.evolve:  construction_heuristic in 【Ant Colony Optimization】
just only change and reture tihis function's mame
only return the implementation of  @funsearch.evolve: construction_heuristic
please sure your is correct python code and just provide function `construction_heuristic` do not include any examples or extraneous functions.
you only need return the relative parameters
Here is the code template, dataset as follows\n
IMPORTANT: Your response should only involve the function  `construction_heuristic`, don't give other code. must visit all nodes!
ATTENTION:@dataclass
only change parameters: num_ants,num_iterations,alpha,beta,rho,Q
must return num_ants,num_iterations,alpha,beta,rho,Q
"""
USER_PROMPT = """
Find the better CVRP solution, try to change @funsearch.evolve:  construction_heuristic
just only change and reture tihis function's mame
only return the implementation of  @funsearch.evolve: construction_heuristic
please sure your is correct python code and just provide function `construction_heuristic` do not include any examples or extraneous functions.
you only need return the relative parameters
Here is the code template, dataset as follows\n
IMPORTANT: Your response should only involve the function  `construction_heuristic`, don't give other code. You must visit all nodes.
Avoid : Exception has occurred: SyntaxError

ATTENTION:@dataclass
"""
USER_PROMPT_GA= """
Find the better CVRP solution, try to change @funsearch.evolve: fitness in 【Genetic Algorithm】
just only change and reture tihis function's mame
only return the implementation of  @funsearch.evolve: fitness
please sure your is correct python code and just provide function `fitness` do not include any examples or extraneous functions.
you only need return the relative parameters
Here is the code template, dataset as follows\n
IMPORTANT: Your response should only involve the function  `fitness`, don't give other code. 
ATTENTION:@dataclass
"""
def read_template_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: can not find {file_path}。")
            return ""
        except Exception as e:
            print(f"Error: unknown {e}。")
            return ""
        
# 需要避免的错误
error_info_arr = ['Exception has occurred: SyntaxError']
# [
#     "AttributeError: 'ConstructionContext' object has no attribute 'distance_matrix'",
#     "unsupported operand type(s) for -: 'tuple' and 'tuple'",
#     "closing parenthesis ']' does not match opening parenthesis '('",
# ]
ERROR_INFO = '\n'.join(error_info_arr)

