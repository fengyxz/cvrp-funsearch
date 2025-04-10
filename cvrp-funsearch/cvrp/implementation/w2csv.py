import csv
import uuid
import os
import logging

# 配置日志格式和日志级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def save_results_to_csv(results, folder="results"):
    """
    保存实验评估结果到 CSV 文件。

    :param results: List[Dict]，每次实验的结果列表，每个结果是一个字典。
    :param folder: str，可选，保存文件的文件夹，默认是 "results" 文件夹。
    """
    # 检查结果是否为空
    if not results:
        logging.warning("No results to save. The results list is empty.")
        return

    # 从结果字典中提取字段名
    fieldnames = results[0].keys()

    # 创建保存文件的文件夹（如果不存在）
    if not os.path.exists(folder):
        os.makedirs(folder)
        logging.info(f"Created directory: {folder}")

    # 使用 uuid 生成文件名
    filename = f"{folder}/results_{uuid.uuid4()}.csv"

    # 开始写入文件
    logging.info("Starting to write results to CSV file...")
    try:
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            # 写入表头
            writer.writeheader()
            # 写入数据行
            writer.writerows(results)

        # 写入完成
        logging.info(f"Results successfully saved to {filename}")
    except Exception as e:
        logging.error(f"An error occurred while writing to the file: {e}")
        raise


# 示例用法
if __name__ == "__main__":
    # 实验的评估结果，每个结果是一个字典
    results = [
        {"is_success": True, "distance": 5.5},
        {"is_success": False, "distance": 12.3},
        {"is_success": True, "distance": 7.8}
    ]

    # 保存结果到 CSV 文件
    save_results_to_csv(results)
