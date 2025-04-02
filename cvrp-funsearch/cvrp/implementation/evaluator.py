
import ast
import copy
import re
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

def _trim_function_body(generated_code: str, generated_func_name: str) -> str:
    """
    解析 LLM 生成的代码，确保只提取函数体。
    """
    code=md_to_source_code(generated_code)
    try:
        module = ast.parse(code)  # 解析 AST
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
    def __init__(self, database, template, function_to_evolve, function_to_run, inputs,timeout_seconds=30,
                ):
        """
        评估 LLM 生成的代码。
        :param database: 代码评估结果数据库
        :param template: 代码模板
        :param function_to_evolve: 目标优化的函数
        :param function_to_run: 要运行的函数
        :param inputs: 测试输入数据
        :param timeout_seconds: 代码运行超时时间
        :param sandbox_class: 沙盒类（用于安全执行代码）
        """
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sand_box.Sandbox()

    def analyse(self, sample,run_func, version_generated):
        """
        评估 LLM 生成的代码，并记录结果。
        """
        new_code = response_to_code(sample, version_generated, self._template, self._function_to_evolve)
        
        scores_per_test = {}
        runs_ok_per_test = {}

        for current_input in self._inputs:
            distance, is_success = self.sand_box.run(program=new_code,function_to_run=run_func,inputs=self._inputs,test_input=current_input,timeout_seconds=500)
            scores_per_test[str(current_input)] = distance
            runs_ok_per_test[str(current_input)] = is_success

