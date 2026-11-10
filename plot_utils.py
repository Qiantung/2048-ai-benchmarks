import json
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 图片清晰度
DPI = 300


# 1. 遗传算法进化曲线（已适配ga_history.json结构）
def plot_ga_history():
    file_path = "ga_history.json"
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件：{file_path}，跳过遗传算法曲线绘制")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        history_list = data["history"]
        generations = [item["generation"] for item in history_list]
        best_fitness = [item["best_fitness"] for item in history_list]
        avg_fitness = [item["average_fitness"] for item in history_list]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, label="种群最优得分", color="#165DFF", linewidth=2)
        plt.plot(generations, avg_fitness, label="种群平均得分", color="#36CFC9", linewidth=2)
        plt.xlabel("进化代数", fontsize=12)
        plt.ylabel("适应度（游戏得分）", fontsize=12)
        plt.title("遗传算法优化启发式权重进化曲线", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("ga_evolution_curve.png", dpi=DPI)
        plt.close()
        print("✅ 遗传算法进化曲线已保存：ga_evolution_curve.png")
    except Exception as e:
        print(f"❌ 遗传算法曲线绘制失败：{str(e)}")


# 2. 监督学习训练曲线（自动匹配CSV字段，解决KeyError）
def plot_supervised_history():
    file_path = "./output_supervised/supervised_history.csv"
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件：{file_path}，跳过监督学习曲线绘制")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        # 自动匹配字段名（兼容多种命名习惯）
        # 匹配轮次
        epoch_key = None
        for key in ["epoch", "epochs", "step", "steps"]:
            if key in fieldnames:
                epoch_key = key
                break
        # 匹配训练损失
        train_loss_key = None
        for key in ["train_loss", "loss_train", "training_loss", "loss"]:
            if key in fieldnames:
                train_loss_key = key
                break
        # 匹配验证损失
        val_loss_key = None
        for key in ["val_loss", "valid_loss", "validation_loss", "test_loss"]:
            if key in fieldnames:
                val_loss_key = key
                break
        # 匹配训练准确率
        train_acc_key = None
        for key in ["train_acc", "acc_train", "train_accuracy", "accuracy_train", "acc"]:
            if key in fieldnames:
                train_acc_key = key
                break
        # 匹配验证准确率
        val_acc_key = None
        for key in ["val_acc", "acc_val", "val_accuracy", "accuracy_val", "test_acc"]:
            if key in fieldnames:
                val_acc_key = key
                break

        # 关键字段缺失则提示
        if not epoch_key or not train_loss_key:
            print("❌ 监督学习CSV字段匹配失败，当前文件包含的字段有：")
            print(fieldnames)
            print("请将上面的字段名发给我，我帮你精准适配")
            return

        # 提取数据
        epochs = []
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        for row in rows:
            epochs.append(int(float(row[epoch_key])))
            train_loss.append(float(row[train_loss_key]))
            if val_loss_key:
                val_loss.append(float(row[val_loss_key]))
            if train_acc_key:
                train_acc.append(float(row[train_acc_key]))
            if val_acc_key:
                val_acc.append(float(row[val_acc_key]))

        # 绘图（有准确率则画双子图，否则只画损失）
        has_acc = train_acc_key is not None and val_acc_key is not None
        if has_acc:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

        # 损失曲线
        ax1.plot(epochs, train_loss, label="训练集", color="#165DFF", linewidth=2)
        if val_loss_key:
            ax1.plot(epochs, val_loss, label="验证集", color="#FF7D00", linewidth=2)
        ax1.set_xlabel("训练轮次", fontsize=12)
        ax1.set_ylabel("交叉熵损失", fontsize=12)
        ax1.set_title("监督学习损失变化曲线", fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)

        # 准确率曲线
        if has_acc:
            ax2.plot(epochs, train_acc, label="训练集", color="#165DFF", linewidth=2)
            ax2.plot(epochs, val_acc, label="验证集", color="#FF7D00", linewidth=2)
            ax2.set_xlabel("训练轮次", fontsize=12)
            ax2.set_ylabel("决策准确率", fontsize=12)
            ax2.set_title("监督学习准确率变化曲线", fontsize=14)
            ax2.legend(fontsize=11)
            ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("supervised_training_curve.png", dpi=DPI)
        plt.close()
        print("✅ 监督学习训练曲线已保存：supervised_training_curve.png")
    except Exception as e:
        print(f"❌ 监督学习曲线绘制失败：{str(e)}")


# 3. DQN强化学习训练曲线
def plot_dqn_history():
    file_path = "./output_dqn_4k/dqn_history.csv"
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件：{file_path}，跳过DQN曲线绘制")
        return

    try:
        episodes = []
        scores = []
        losses = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                scores.append(float(row["score"]))
                losses.append(float(row["loss"]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        window = 50

        # 得分曲线（滑动平均）
        smoothed_scores = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax1.plot(episodes, scores, color="#E8F3FF", linewidth=0.8, label="单局得分")
        ax1.plot(episodes[window-1:], smoothed_scores, color="#165DFF", linewidth=2, label=f"{window}局滑动平均")
        ax1.set_xlabel("训练回合", fontsize=12)
        ax1.set_ylabel("游戏得分", fontsize=12)
        ax1.set_title("DQN强化学习得分变化曲线", fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)

        # 损失曲线（滑动平均）
        smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax2.plot(episodes, losses, color="#FFF1E8", linewidth=0.8, label="单步损失")
        ax2.plot(episodes[window-1:], smoothed_losses, color="#FF7D00", linewidth=2, label=f"{window}步滑动平均")
        ax2.set_xlabel("训练回合", fontsize=12)
        ax2.set_ylabel("MSE损失值", fontsize=12)
        ax2.set_title("DQN强化学习损失变化曲线", fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("dqn_training_curve.png", dpi=DPI)
        plt.close()
        print("✅ DQN训练曲线已保存：dqn_training_curve.png")
    except Exception as e:
        print(f"❌ DQN曲线绘制失败：{str(e)}")


# 4. 四种AI策略对比柱状图
def plot_comparison():
    file_path = "comparison_result.csv"
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件：{file_path}，跳过对比柱状图绘制")
        return

    try:
        methods = []
        avg_scores = []
        avg_max_tiles = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                methods.append(row["方法"])
                avg_scores.append(float(row["平均得分"]))
                avg_max_tiles.append(float(row["平均最大方块"]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = ["#91CBFF", "#165DFF", "#36CFC9", "#FF7D00"]

        # 平均得分对比
        bars1 = ax1.bar(methods, avg_scores, color=colors, edgecolor="white", width=0.6)
        ax1.set_ylabel("平均得分", fontsize=12)
        ax1.set_title("四种AI策略平均得分对比", fontsize=14)
        ax1.grid(axis='y', alpha=0.3)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=11)

        # 平均最大方块对比
        bars2 = ax2.bar(methods, avg_max_tiles, color=colors, edgecolor="white", width=0.6)
        ax2.set_ylabel("平均最大方块", fontsize=12)
        ax2.set_title("四种AI策略平均最大方块对比", fontsize=14)
        ax2.grid(axis='y', alpha=0.3)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        plt.savefig("ai_comparison_bar.png", dpi=DPI)
        plt.close()
        print("✅ AI对比柱状图已保存：ai_comparison_bar.png")
    except Exception as e:
        print(f"❌ 对比柱状图绘制失败：{str(e)}")


if __name__ == "__main__":
    print("开始生成所有实验图表...\n")
    plot_ga_history()
    plot_supervised_history()
    plot_dqn_history()
    plot_comparison()
    print("\n🎉 所有图表生成任务完成，成功生成的图片已保存到当前目录")