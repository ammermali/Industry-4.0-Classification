import pandas as pd
import matplotlib.pyplot as plt
import os

# Components that converts the history.csv file into an unified plot for comparison.


def plot_models(experiments):
    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    fig.suptitle('Models Comparison & Performance Analysis', fontsize=18, y=0.98)
    summary_table = []
    for exp in experiments:
        name = exp['name']
        csv_path = f'logs/{name}/history.csv'
        if not os.path.exists(csv_path):
            print(f"Data for {name} not found in {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        epochs = range(1, len(df) + 1)
        line = ax1.plot(epochs, df['loss'], label=f'{name} Train', linewidth=2)
        color = line[0].get_color()
        ax1.plot(epochs, df['val_loss'], color = color, label=f'{name} Val', linewidth=2, linestyle='--')
        ax2.plot(epochs, df['accuracy'], color = color, label=f'{name} Train', linewidth=2)
        ax2.plot(epochs, df['val_accuracy'], color = color, label=f'{name} Val', linewidth=2, linestyle='--')
        if 'epoch_time' in df.columns:
            duration = df['epoch_time'].sum()
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            time_str = f"{minutes}m {seconds}s"
            summary_table.append([name, time_str])
        else:
            summary_table.append([name, "N/A"])

    ax1.set_title('Loss Comparison', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()
    ax2.set_title('Accuracy Comparison', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()

    ax3.axis('off')
    if summary_table:
        table = ax3.table(
            cellText=summary_table,
            colLabels=['Experiment', 'Total Time'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.1, 2.5)
    ax3.set_title('Time for Training', fontsize=14, pad=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = 'logs/models_comparison.png'
    plt.savefig(save_path)