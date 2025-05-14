import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Function to read JSONL log files
def read_logs(file_paths):
    logs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue
        log_data = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    # Skip empty lines
                    if line.strip():
                        try:
                            log_entry = json.loads(line.strip())
                            log_data.append(log_entry)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse line in {file_path}: {line.strip()}. Error: {e}")
            logs.append(log_data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return logs

# Function to extract data for plotting
def extract_data(logs, metric):
    data = []
    for i, log in enumerate(logs):
        epochs = [entry['epoch'] for entry in log]
        values = [entry[metric] for entry in log]
        data.append({'epochs': epochs, 'values': values, 'model': f'Model {i+1}'})
    return data

# Main plotting function
def plot_metrics(log_files, metrics, output_html='model_comparison.html'):
    # Read logs
    logs = read_logs(log_files)
    if not logs:
        print("No valid log files found.")
        return

    # Create a subplot
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Model Metrics Comparison'])

    # Metrics to plot
    # metrics = ['test_acc1', 'test_acc5', 'test_loss', 'test_ema_loss']

    # Initialize visibility: only the first metric is visible initially
    for metric_idx, metric in enumerate(metrics):
        metric_data = extract_data(logs, metric)
        visibility = (metric_idx == 0)  # Only first metric is visible initially

        for model_data in metric_data:
            fig.add_trace(
                go.Scatter(
                    x=model_data['epochs'],
                    y=model_data['values'],
                    mode='lines+markers',
                    name=f"{model_data['model']} - {metric}",
                    visible=visibility,
                    legendgroup=metric,
                    showlegend=(metric_idx == 0)  # Show legend only for first metric traces
                )
            )

    # Create dropdown buttons
    buttons = []
    for i, metric in enumerate(metrics):
        # Create visibility array: True for traces of current metric, False otherwise
        visibility = [False] * (len(metrics) * len(logs))
        start_idx = i * len(logs)
        end_idx = start_idx + len(logs)
        visibility[start_idx:end_idx] = [True] * len(logs)

        buttons.append(
            dict(
                label=metric,
                method='update',
                args=[
                    {'visible': visibility},
                    {'title': f'Comparison of {metric}'}
                ]
            )
        )

    # Update layout with dropdown
    fig.update_layout(
        title='Comparison of test_acc1',
        xaxis_title='Epoch',
        yaxis_title='Value',
        updatemenus=[
            dict(
                buttons=buttons,
                direction='down',
                showactive=True,
                x=0.1,
                xanchor='left',
                y=1.1,
                yanchor='top'
            )
        ],
        showlegend=True
    )

    # Save to HTML
    fig.write_html(output_html)
    print(f"Plot saved to {output_html}")
# Example usage
if __name__ == '__main__':
    log_files = [
        '/home/zengshimao/code/RMT-main/rankControl/Stable_V_rank/log.txt',
        '/home/zengshimao/code/RMT-main/rankControl/rankTestOutput/log.txt',
        '/home/zengshimao/code/RMT-main/rankControl/stable_rank_all/log.txt'
    ]
    metrics = ['test_acc1', 'test_acc5', 'test_loss','train_loss', 'test_ema_loss', 'test_ema_acc1', 'test_ema_acc5']
    plot_metrics(log_files, metrics)