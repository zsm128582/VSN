import json
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact

def read_logs(file_path):
    logs = []
    with open(file_path, 'r') as file:
        for line in file:
            log = json.loads(line)
            logs.append(log)
    return logs

def plot_metric(logs_list, metric):
    fig, ax = plt.subplots(figsize=(10, 6))
    for logs in logs_list:
        epochs = [log['epoch'] for log in logs]
        values = [log[metric] for log in logs]
        ax.plot(epochs, values, marker='o', label=f'Model {logs_list.index(logs) + 1}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} over Epochs')
    ax.legend()
    plt.show()

# 读取三个模型的log文件
logs1 = read_logs('/home/zengshimao/code/RMT-main/rankControl/stable_rank_all/log.txt')
logs2 = read_logs('/home/zengshimao/code/RMT-main/rankControl/rankTestOutput/log.txt')
logs3 = read_logs('/home/zengshimao/code/RMT-main/RMT_log/rmt_t.txt')

# 使用interact创建交互式绘图
interact(plot_metric, logs_list=[[logs1, logs2, logs3]], metric=['test_acc1', 'test_acc5', 'test_loss', 'test_ema_loss', 'test_ema_acc1', 'test_ema_acc5']);