import os
import time
import csv
from collections import defaultdict


class CostRecord:
    records_by_experiment = defaultdict(lambda: defaultdict(list))
    current_experiment = "default"  # 預設實驗名稱

    def __init__(self, function_name):
        self.function_name = function_name
        self.start_time = None

    @classmethod
    def set_experiment(cls, experiment_name):
        """設定當前實驗的名稱"""
        cls.current_experiment = experiment_name

    def __enter__(self):
        # 記錄開始時間
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        CostRecord.records_by_experiment[CostRecord.current_experiment][self.function_name].append(elapsed_time)

    @staticmethod
    def export_to_csv(file_path='timing_records.csv'):
        """
        将每个实验名称和函数名称的计时数据写入 CSV 文件。

        Parameters:
            file_path (str): CSV 文件路径
        """
        file_exists = os.path.exists(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(
                    ['Experiment Name', 'Function Name', 'Average Time', 'Average FPS', 'Execution Times (seconds)'])

            for experiment_name, functions in CostRecord.records_by_experiment.items():
                for function_name, execution_times in functions.items():
                    avg_time = sum(execution_times) / len(execution_times)
                    avg_fps = 1.0 / avg_time if avg_time > 0.001 else 0.001
                    writer.writerow([experiment_name, function_name, avg_time, avg_fps] + execution_times)


if __name__ == '__main__':
    def some_function():
        with CostRecord("some_function"):
            time.sleep(1)  # 模擬函數執行


    def another_function():
        with CostRecord("another_function"):
            time.sleep(2)  # 模擬另一個函數執行


    # 設定實驗名稱為 "Experiment 1"
    CostRecord.set_experiment("Experiment 1")
    some_function()

    # 設定實驗名稱為 "Experiment 2"
    CostRecord.set_experiment("Experiment 2")
    another_function()

    # 再次切換回 "Experiment 1"
    CostRecord.set_experiment("Experiment 1")
    another_function()

    # 程式結束時匯出計時記錄
    # import atexit
    # def export_to_csv():
    #     CostRecord.export_to_csv('timing_records.csv')
    # atexit.register(export_to_csv)

    # 程式結束時匯出計時記錄
    import atexit

    atexit.register(CostRecord.export_to_csv)
