import json
import matplotlib.pyplot as plt


def read_json_and_plot(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        block_data = []
        layer_boundaries = []
        current_block_index = -1

        for layer in data.values():
            layer_block_count = len(layer)
            current_block_index += layer_block_count
            layer_boundaries.append(current_block_index)
            for block in layer.values():
                block_data.append(block)

        x = list(range(len(block_data)))
        Q_values = [block['Q'] for block in block_data]
        K_values = [block['K'] for block in block_data]
        V_values = [block['V'] for block in block_data]

        plt.plot(x, Q_values, label='Q')
        plt.plot(x, K_values, label='K')
        plt.plot(x, V_values, label='V')

        for boundary in layer_boundaries[:-1]:
            plt.axvline(x=boundary, color='r', linestyle='--', label='Layer Boundary' if boundary == layer_boundaries[0] else "")

        plt.xlabel('Block Number')
        plt.ylabel('Value')
        plt.title('Q, K, V Metrics Trend')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig("stable_rank.png")

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print("错误: 无法解析 JSON 文件。")
    except KeyError:
        print("错误: JSON 文件中缺少 Q、K 或 V 键。")


if __name__ == "__main__":
    file_path = '/home/zengshimao/code/RMT-main/rankControl/stable_rank_all/effective_rank.json'  # 请替换为实际的 JSON 文件路径
    read_json_and_plot(file_path)
    