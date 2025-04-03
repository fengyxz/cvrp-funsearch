import multiprocessing
import logging



# 定义 Sandbox 类，用于安全执行 LLM 生成的代码
class Sandbox():
    """
    Sandbox（沙盒）用于安全执行 LLM 生成的代码，并返回计算结果。

    主要功能：
    1) 防止代码执行恶意操作（如访问互联网、占用过多 RAM）。
    2) 设定超时时间，防止死循环或无限执行的情况。
    3) 确保 CVRP 约束被严格执行，非法解会被拒绝。
    """

    def __init__(self, verbose=True):
        """
        初始化 Sandbox。

        :param verbose: 是否打印详细的评估信息，默认为 False。
        """
        self._verbose = verbose
    def run(
            self,
            program: str,
            function_to_run: str,  # 需要运行的函数名
            input: any,  # 输入数据集
            timeout_seconds: int,  # 允许执行的最长时间（超时即失败）
            **kwargs  # 额外参数
    ) -> tuple[bool, float, any]:
        """
        运行 `function_to_run(test_input)` 并返回结果。
        若代码执行失败（如超时、错误或不符合 CVRP 约束），则返回 (None, False)。

        :param program: 需要执行的 Python 代码（由 LLM 生成）。
        :param function_to_run: 代码中要执行的函数名，例如 'evaluate'。
        :param test_input: 当前测试输入的 key（用于从数据集中获取数据）。
        :param timeout_seconds: 允许的最大执行时间（秒）。
        :return: (score, success)，如果执行成功，则返回计算得分和 True，否则返回 (None, False)。
        """

        dataset = input  # 提取当前测试数据
        try:
            result_queue = multiprocessing.Queue()  # 用于存储执行结果
            process = multiprocessing.Process(
                target=self._compile_and_run_function,
                args=(program, function_to_run, dataset, result_queue)
            )

            process.start()  # 启动进程
            process.join(timeout=timeout_seconds)  # 等待进程结束，若超时则终止

            if process.is_alive():
                # 若超时，则终止进程并返回非法解
                process.terminate()
                process.join()
                return None
            if not result_queue.empty():
                results = result_queue.get_nowait()  # 获取计算结果
                return results
            return None
        except Exception:
            logging.error("代码执行出错！",Exception.with_traceback)
            return None  # 代码执行出错

    def _compile_and_run_function(self, program, function_to_run, dataset, result_queue):
        """
        编译并运行 LLM 生成的代码。
        运行 LLM 生成的 `function_to_run`，并返回计算结果。

        :param program: 需要执行的 Python 代码（由 LLM 生成）。
        :param function_to_run: 代码中要执行的函数名。
        :param dataset: 当前测试数据集。
        :param result_queue: 用于存储执行结果的队列（避免进程间数据共享问题）。
        """
        try:
            all_globals_namespace = {}  # 存储全局变量、函数和类的命名空间
            exec(program, all_globals_namespace)  # 执行 LLM 生成的代码
            function_to_run = all_globals_namespace[function_to_run]  # 获取目标函数指针
            # CVRP 约束检查
            results = function_to_run(dataset)  # 运行路径规划算法，返回解
            result_queue.put(results)  # 返回评估值
        except Exception:
            result_queue.put(('FAIL',"",[]))  # 代码执行失败，返回非法解
