import ast
import copy
from abc import ABC, abstractmethod
from time import time
import code_manipulation


# 沙盒机制：用于安全地执行代码，防止恶意代码执行
class Sandbox(ABC):
    @abstractmethod
    def run(self, program: str, function_to_run: str, function_to_evolve: str,
            inputs: any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[any, bool]:
        """
        在沙盒环境中运行给定的代码。
        :param program: 需要运行的完整代码
        :param function_to_run: 要调用的函数名
        :param function_to_evolve: 目标进化的函数名
        :param inputs: 测试输入数据
        :param test_input: 当前测试输入
        :param timeout_seconds: 运行超时时间
        :return: (测试输出, 是否成功运行)
        """
        pass


# 修剪 LLM 生成的代码，确保只包含函数体

def _trim_function_body(generated_code: str) -> str:
    """
    解析 LLM 生成的代码，确保只提取函数体。
    """
    function_header = "def fake_function_header():"  # 临时函数头
    while True:
        try:
            code = function_header + "\n" + generated_code  # 构造完整代码
            module = ast.parse(code)  # 解析 AST
            break  # 成功解析则跳出循环
        except SyntaxError:
            generated_code = "\n".join(generated_code.split("\n")[:-1])  # 移除最后一行，尝试修复语法错误

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "fake_function_header":
            end_lineno = node.end_lineno  # 获取函数体的结束行号
            break

    return "\n".join(generated_code.split("\n")[:end_lineno - 1])  # 仅保留函数体


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


# 代码评估器
class Evaluator:
    def __init__(self, database, template, function_to_evolve, function_to_run, inputs, timeout_seconds=30,
                 sandbox_class=Sandbox):
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
