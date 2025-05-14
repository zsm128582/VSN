import json
import matplotlib.pyplot as plt


def read_json_lines(file_path):
    data_list = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError:
                    print(f"解析 JSON 时出错，跳过行: {line}")
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    return data_list


def plot_metrics(data_list):
    epochs = [data["epoch"] for data in data_list]
    loss_metrics = [
        "train_loss",
        "test_loss",
        "test_ema_loss"
    ]
    acc_metrics = [
        "test_acc1",
        "test_acc5",
        "test_ema_acc1",
        "test_ema_acc5"
    ]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 绘制 loss 指标到左侧坐标轴
    for metric in loss_metrics:
        values = [data[metric] for data in data_list]
        ax1.plot(epochs, values, label=metric)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    # 创建右侧坐标轴
    ax2 = ax1.twinx()

    # 绘制 acc 指标到右侧坐标轴
    for metric in acc_metrics:
        values = [data[metric] for data in data_list]
        ax2.plot(epochs, values, label=metric, linestyle='--')

    ax2.set_ylabel('Accuracy', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    plt.title('Metrics over Epochs')
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    file_path = '/home/zengshimao/code/RMT-main/rankControl/effective_ranks_svdControl.json'  # 请替换为实际的文件路径
    data_list = read_json_lines(file_path)
    if data_list:
        plot_metrics(data_list)
    