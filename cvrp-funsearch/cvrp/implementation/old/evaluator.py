import ast
import copy
from abc import ABC, abstractmethod
import re
from time import time
import code_manipulation

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
    module = ast.parse(code)  # 解析 AST
    try:
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == generated_func_name:
                end_lineno = node.end_lineno  # 获取函数体的结束行号
                break
        return "\n".join(code.split("\n")[:end_lineno - 1])  # 仅保留函数体
    except():
        KeyError
        


# 生成完整可运行的程序

def _sample_to_program(generated_code, version_generated, template, function_to_evolve):
    """
    结合模板代码，生成完整可运行的程序。
    """
    trimmed_function = _trim_function_body(generated_code)  # 修剪函数体
    if version_generated:
        trimmed_function = code_manipulation.rename_function_calls(trimmed_function, version_generated,
                                                                   function_to_evolve)  # 重命名函数
    program = copy.deepcopy(template)  # 复制模板代码
    new_function = code_manipulation.replace_function(program, function_to_evolve, trimmed_function)  # 替换目标函数
    return new_function, program

sand_box = sand_box.Sandbox()
# 代码评估器
class Evaluator:
    def __init__(self, database, template, function_to_evolve, function_to_run, inputs, timeout_seconds=30,
                 sandbox_class=sandbox):
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
        self._sandbox = sandbox_class()

    def analyse(self, sample, island_id, version_generated, **kwargs):
        """
        评估 LLM 生成的代码，并记录结果。
        """
        new_function, program = _sample_to_program(sample, version_generated, self._template, self._function_to_evolve)
        scores_per_test = {}
        runs_ok_per_test = {}

        for current_input in self._inputs:
            test_output, runs_ok = self._sandbox.run(
                program, self._function_to_run, self._function_to_evolve, self._inputs, current_input,
                self._timeout_seconds
            )
            if test_output is not None and isinstance(test_output, (int, float)) and runs_ok:
                scores_per_test[current_input] = test_output
                runs_ok_per_test[current_input] = runs_ok

        if scores_per_test:
            self._database.register_program(
                program=program, scores_per_test=scores_per_test, runs_ok_per_test=runs_ok_per_test,
                island_id=island_id, version_generated=version_generated, **kwargs
            )
        else:
            self._database.profiler.add_execution(island_id, version_generated, execution_time=time(),
                                                  failed=True)  # 记录失败
