import multiprocessing
import evaluator
import evaluator_accelerate


# 定义 Sandbox 类，用于安全执行 LLM 生成的代码
class Sandbox(evaluator.Sandbox):
    """
    Sandbox（沙盒）用于安全执行 LLM 生成的代码，并返回计算结果。

    主要功能：
    1) 防止代码执行恶意操作（如访问互联网、占用过多 RAM）。
    2) 设定超时时间，防止死循环或无限执行的情况。
    3) 确保 CVRP 约束被严格执行，非法解会被拒绝。
    """

    def __init__(self, verbose=False, numba_accelerate=True):
        """
        初始化 Sandbox。

        :param verbose: 是否打印详细的评估信息，默认为 False。
        :param numba_accelerate: 是否使用 numba 加速计算（部分 numpy 函数如 np.piecewise() 不支持 numba）。
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(
            self,
            program: str,
            function_to_run: str,  # 需要运行的函数名
            function_to_evolve: str,  # 需要加速的函数名
            inputs: any,  # 输入数据集
            test_input: str,  # 当前测试实例
            timeout_seconds: int,  # 允许执行的最长时间（超时即失败）
            **kwargs  # 额外参数
    ) -> tuple[any, bool]:
        """
        运行 `function_to_run(test_input)` 并返回结果。
        若代码执行失败（如超时、错误或不符合 CVRP 约束），则返回 (None, False)。

        :param program: 需要执行的 Python 代码（由 LLM 生成）。
        :param function_to_run: 代码中要执行的函数名，例如 'evaluate'。
        :param function_to_evolve: 需要优化的函数名，可用于 numba 加速。
        :param inputs: 训练/测试数据集。
        :param test_input: 当前测试输入的 key（用于从数据集中获取数据）。
        :param timeout_seconds: 允许的最大执行时间（秒）。
        :return: (score, success)，如果执行成功，则返回计算得分和 True，否则返回 (None, False)。
        """

        dataset = inputs[test_input]  # 提取当前测试数据
        try:
            result_queue = multiprocessing.Queue()  # 用于存储执行结果
            process = multiprocessing.Process(
                target=self._compile_and_run_function,
                args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
            )

            process.start()  # 启动进程
            process.join(timeout=timeout_seconds)  # 等待进程结束，若超时则终止

            if process.is_alive():
                # 若超时，则终止进程并返回非法解
                process.terminate()
                process.join()
                return None, False

            if not result_queue.empty():
                results = result_queue.get_nowait()  # 获取计算结果
                score, success = results

                # CVRP 约束检查
                if success:
                    if not _is_valid_cvrp_solution(score, dataset):
                        return None, False  # 解不合法

                    # 计算总路径长度（目标函数）
                    total_distance = _compute_total_distance(score, dataset)
                    return total_distance, True  # 返回目标值和成功标志

            return None, False
        except:
            return None, False  # 代码执行出错，返回非法解

    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, dataset, numba_accelerate,
                                  result_queue):
        """
        编译并运行 LLM 生成的代码。

        1) 先给 `function_to_evolve` 加上 numba 加速（如果启用）。
        2) 运行 LLM 生成的 `function_to_run`，并返回计算结果。
        3) 若结果不符合 CVRP 约束，则返回非法解。

        :param program: 需要执行的 Python 代码（由 LLM 生成）。
        :param function_to_run: 代码中要执行的函数名。
        :param function_to_evolve: 需要优化的函数名，可用于 numba 加速。
        :param dataset: 当前测试数据集。
        :param numba_accelerate: 是否启用 numba 加速。
        :param result_queue: 用于存储执行结果的队列（避免进程间数据共享问题）。
        """
        try:
            if numba_accelerate:
                # 在 `function_to_evolve` 上添加 numba 加速
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )

            all_globals_namespace = {}  # 存储全局变量、函数和类的命名空间
            exec(program, all_globals_namespace)  # 执行 LLM 生成的代码
            function_to_run = all_globals_namespace[function_to_run]  # 获取目标函数指针

            results = function_to_run(dataset)  # 运行路径规划算法，返回解

            # CVRP 约束检查
            if not _is_valid_cvrp_solution(results, dataset):
                result_queue.put((None, False))  # 解不合法
                return

            # 计算总路径长度
            total_distance = _compute_total_distance(results, dataset)
            result_queue.put((total_distance, True))  # 返回评估值

        except Exception:
            result_queue.put((None, False))  # 代码执行失败，返回非法解


def _compute_total_distance(solution, dataset):
    """
    计算 CVRP 解的总路径长度。

    :param solution: 车辆路径方案 {vehicle_id: [route]}。
    :param dataset: 数据集，包含 'distance_matrix'（距离矩阵）。
    :return: 该解的总路径长度。
    """
    total_distance = 0

    for route in solution.values():
        if len(route) < 2:
            continue  # 只访问一个点，无需计算路径长度

        for i in range(len(route) - 1):
            total_distance += dataset["distance_matrix"][route[i]][route[i + 1]]

        # 加上返回起点（假设所有车辆从同一仓库出发）
        total_distance += dataset["distance_matrix"][route[-1]][route[0]]

    return total_distance


def _is_valid_cvrp_solution(solution, dataset):
    """
    检查 CVRP（容量约束车辆路径问题）的解是否满足以下约束：
    1) 每个客户必须被访问一次，且仅访问一次。
    2) 每辆车的总负载不能超过其容量限制。
    3) 没有重复访问的路径。

    :param solution: 由路径规划算法生成的解，假设格式为 {vehicle_id: [route]}。
    :param dataset: 提供客户需求量（demand）和车辆容量（vehicle_capacity）。
    :return: 如果解合法返回 True，否则返回 False。
    """
    if not isinstance(solution, dict):  # 确保解的格式正确
        return False

    visited_customers = set()  # 记录所有被访问的客户
    vehicle_loads = {}  # 记录每辆车的总负载

    for vehicle_id, route in solution.items():
        if len(set(route)) != len(route):  # 约束一：检查是否有重复访问的路径
            return False

        total_demand = sum(dataset["demand"][customer] for customer in route if customer in dataset["demand"])
        if total_demand > dataset["vehicle_capacity"]:  # 约束二：检查车辆是否超载
            return False

        visited_customers.update(route)  # 记录访问的客户
        vehicle_loads[vehicle_id] = total_demand

    all_customers = set(dataset["demand"].keys())  # 数据集中所有客户
    if visited_customers != all_customers:  # 约束三：确保所有客户都被访问一次
        return False

    return True  # 通过所有检查


if __name__ == '__main__':
    # class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    # config = config.Config(samples_per_prompt=4, evaluate_timeout_seconds=30)
    # global_max_sample_num = 10  # if it is set to None, funsearch will execute an endless loop
    # funsearch.main(
    #     specification=specification,
    #     inputs=bin_packing_or3,
    #     config=config,
    #     max_sample_nums=global_max_sample_num,
    #     class_config=class_config,
    #     log_dir='../logs/funsearch_llm_api'
    # )
    pass