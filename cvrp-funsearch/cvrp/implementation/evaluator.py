
import ast
import copy
import re
import time
import sand_box


# 修剪 LLM 生成的代码，确保只包含函数体
def md_to_source_code(md_text):
    # 定义正则表达式模式来匹配 Python 代码块
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    match = pattern.search(md_text)
    if match:
        # 提取代码块内容并去除首尾空格
        return match.group(1).strip()
    return None

def _trim_function_body(generated_code: str) -> str:
    """
    解析 LLM 生成的代码，确保只提取函数体。
    """
    code=md_to_source_code(generated_code)
    try:
        return "\n".join(code.split("\n"))  # 仅保留函数体
    except():
        KeyError
        
def response_to_code(generated_response:str, template:str, function_to_evolve:str):
    """
    generated_response: 生成的回复;
    template: 使用的模版代码;
    function_to_evolve: 需要替换的函数名称;
    结合模板代码，生成完整可运行的程序。
    """
    trimmed_function = _trim_function_body(generated_response)  # 修剪函数体
    program = copy.deepcopy(template)  # 复制模板代码
    new_code = replace_function_by_name(program, function_to_evolve, trimmed_function)  # 替换目标函数
    return new_code

def replace_function_by_name(source_code, old_function_name, new_function_str):
    # 解析源代码为抽象语法树
    print(source_code)
    tree = ast.parse(source_code)
    new_function_ast =  ast.parse(new_function_str)
    # 定义一个访问者类，用于遍历抽象语法树
    class FunctionReplacer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # 如果当前节点是函数定义，并且函数名与要替换的函数名相同
            if node.name == old_function_name:
                # 返回新的函数定义节点来替换原节点
                return new_function_ast
            # 否则保留该节点
            return self.generic_visit(node)

    # 创建访问者实例
    replacer = FunctionReplacer()
    # 对抽象语法树进行转换
    new_tree = replacer.visit(tree)

    # 将修改后的抽象语法树转换回 Python 代码
    new_code = ast.unparse(new_tree)
    return new_code

# 代码评估器
class Evaluator:
    def __init__(self, template, function_to_evolve, function_to_run, input,timeout_seconds=300,
                ):
        """
        评估 LLM 生成的代码。
        :param template: 代码模板
        :param function_to_evolve: 目标优化的函数
        :param function_to_run: 要运行的函数
        :param input: 测试输入数据
        :param timeout_seconds: 代码运行超时时间
        :param sandbox_class: 沙盒类（用于安全执行代码）
        """
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._input = input
        self._timeout_seconds = timeout_seconds

    def analyse(self, sample):
        """
        评估 LLM 生成的代码，并记录结果。
        """
        start_time = time.time()  # 记录开始时间
        sandbox = sand_box.Sandbox()
        data = self._input

        new_code = response_to_code(sample,self._template, self._function_to_evolve)
        
        scores_per_test = {}
        runs_ok_per_test = {}

        # 获得生成的route
        routes = sandbox.run(program=new_code,function_to_run=self._function_to_run,input=data,timeout_seconds=self._timeout_seconds)
        total_distance = 0.0
        distance_matrix = data['distance_matrix']
        depot = data['depot']
        demand = data['demand']
        capacity = data['vehicle_capacity']
        all_nodes = set(range(len(demand)))
        all_nodes.remove(depot)  # 移除仓库点，因为仓库点会多次出现
        visited_nodes = set()
        
         #评估解决方案并返回状态和总行驶距离
        try:
            for i, route in enumerate(routes):
                # 起点和终点都要是depot
                if len(route) < 2 or route[0] != depot or route[-1] != depot:
                    raise ValueError(f"Invalid route {i}: {route} - must start and end at depot")

                route_distance = 0.0
                route_demand = 0
                for j in range(len(route) - 1):
                    from_node = route[j]
                    to_node = route[j + 1]
                    route_distance += distance_matrix[from_node][to_node]
                    if to_node != depot:  # depot仓库需求为0
                        route_demand += demand[to_node]
                # 是否超出容量
                if route_demand > capacity:
                    raise ValueError(f"Route {i} overloaded: {route_demand}/{capacity}")
                total_distance += route_distance
                print(f"Vehicle {i} route: {route}")
            
            # 检查是否所有点都被访问
            if visited_nodes != all_nodes:
                missing_nodes = all_nodes - visited_nodes
                raise ValueError(f"Not all nodes are visited. Missing nodes: {missing_nodes}")
            end_time = time.time()  # 记录结束时间
            run_time = end_time - start_time  # 计算运行时间
            print(f"Total  Distance: {route_distance:.2f}, Load: {route_demand}/{capacity}\n")
            print(f"Vehicles used: {sum(1 for r in routes if len(r) > 2)}/{len(routes)}")  # 排除空路线
            print(f"Run time: {run_time:.4f} seconds")  # 打印运行时间
            return True, total_distance, route, run_time
        except ValueError as e:
            end_time = time.time()  # 记录结束时间
            run_time = end_time - start_time  # 计算运行时间
            print(f"Evaluation failed: {e}")
            print(f"Run time: {run_time:.4f} seconds")  # 打印运行时间
            return False, "", [], run_time
   

