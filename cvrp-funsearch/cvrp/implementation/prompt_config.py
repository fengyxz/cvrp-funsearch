  
# 用户指令
USER_PROMPT = """
Find the better CVRP solution, try to change @funsearch.evolve:  construction_heuristic
just only change and reture tihis function's mame
only return the implementation of  @funsearch.evolve: construction_heuristic
please sure your is correct python code and just provide function `construction_heuristic` do not include any examples or extraneous functions.

Here is the code template, dataset as follows\n
IMPORTANT: Your response should only involve the function  `construction_heuristic`, don't give other code.
ATTENTION:@dataclass
class ConstructionContext:
    depot: int
    candidate: int
    distance: float
    demand: int
    vehicle_load: int
    vehicle_capacity: int
    locations: np.ndarray
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
        
# 代码模版
CODE_TEMPLATE = read_template_file('cvrp-funsearch/cvrp/spec/simple_code_template.txt')
# 需要避免的错误
error_info_arr = [
    "AttributeError: 'ConstructionContext' object has no attribute 'distance_matrix'",
    "unsupported operand type(s) for -: 'tuple' and 'tuple'"
]
ERROR_INFO = '\n'.join(error_info_arr)

