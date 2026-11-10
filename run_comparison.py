"""
四种2048 AI策略性能对比脚本
1. 基础启发式AI（默认权重）
2. 进化优化后的启发式AI（遗传算法调优权重）
3. 监督学习AI（CNN模仿学习）
4. DQN强化学习AI（预训练+微调）
"""

import json
import csv
import torch
from utils import (
    evaluate_ai,
    ACTION_TO_ID,
    flatten_board,
    get_legal_actions,
)
from heuristic_ai import HeuristicAI
from cnn_model import CNNPolicyNetwork
from dqn_agent import DQNAgent, DQNConfig


# ===================== 接口统一包装类（适配evaluate_ai的调用规范） =====================
class HeuristicAIWrapper:
    """包装启发式AI，统一select_action接口，返回int类型动作ID"""
    def __init__(self, weights=None):
        self.ai = HeuristicAI(weights)

    def select_action(self, game):
        # 调用原生方法获取方向字符串，转为int格式适配游戏接口
        dir_action = self.ai.choose_action(game)
        if dir_action is None:
            legal = get_legal_actions(game)
            return legal[0] if legal else 0
        return ACTION_TO_ID[dir_action]


class SupervisedAIWrapper:
    """包装监督学习CNN模型，统一select_action接口"""
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNPolicyNetwork().to(self.device)
        # 加载监督学习预训练权重
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def select_action(self, game):
        # 提取棋盘log2特征，与训练时输入格式保持一致
        state = flatten_board(game.get_board())
        with torch.no_grad():
            tensor = torch.tensor(state.reshape(1, -1), device=self.device, dtype=torch.float32)
            logits = self.model(tensor)
            # 返回置信度最高的动作ID
            return int(logits.argmax(dim=1).item())


class DQNAIWrapper:
    """包装DQN强化学习智能体，关闭随机探索，纯贪心测试"""
    def __init__(self, model_path):
        self.agent = DQNAgent(DQNConfig())
        self.agent.load(model_path)
        # 测试模式：完全关闭随机探索，输出模型真实决策水平
        self.agent.epsilon = 0.0

    def select_action(self, game):
        # 提取棋盘状态
        state = flatten_board(game.get_board())
        # 获取当前合法动作，屏蔽非法方向
        legal_actions = get_legal_actions(game)
        # 关闭训练模式，纯贪心决策
        return self.agent.select_action(state, legal_actions=legal_actions, training=False)


# ===================== 主流程：依次评估四种AI =====================
if __name__ == "__main__":
    # 评估配置：每个AI跑10局取平均，数值越大结果越稳定，耗时越长
    EVAL_GAMES = 10

    # 1. 基础启发式AI（默认人工权重）
    print("=" * 60)
    print("正在评估：基础启发式AI")
    base_ai = HeuristicAIWrapper()
    base_result = evaluate_ai(base_ai, games=EVAL_GAMES)
    print(f"平均得分: {base_result['avg_score']:.1f} | 平均最大方块: {base_result['avg_max_tile']:.1f}")

    # 2. 进化优化后的启发式AI（遗传算法最优权重）
    print("\n" + "=" * 60)
    print("正在评估：进化优化启发式AI")
    with open("best_weights.json", "r", encoding="utf-8") as f:
        best_weights = json.load(f)
    ga_ai = HeuristicAIWrapper(best_weights)
    ga_result = evaluate_ai(ga_ai, games=EVAL_GAMES)
    print(f"平均得分: {ga_result['avg_score']:.1f} | 平均最大方块: {ga_result['avg_max_tile']:.1f}")

    # 3. 监督学习AI（CNN模仿启发式策略）
    print("\n" + "=" * 60)
    print("正在评估：监督学习AI")
    sl_ai = SupervisedAIWrapper("./output_supervised/supervised_policy.pt")
    sl_result = evaluate_ai(sl_ai, games=EVAL_GAMES)
    print(f"平均得分: {sl_result['avg_score']:.1f} | 平均最大方块: {sl_result['avg_max_tile']:.1f}")

    # 4. DQN强化学习AI（监督预训练+强化微调）
    print("\n" + "=" * 60)
    print("正在评估：DQN强化学习AI")
    dqn_ai = DQNAIWrapper("./output_dqn_4k/dqn_policy.pt")
    dqn_result = evaluate_ai(dqn_ai, games=EVAL_GAMES)
    print(f"平均得分: {dqn_result['avg_score']:.1f} | 平均最大方块: {dqn_result['avg_max_tile']:.1f}")

    # 保存对比结果到CSV，可直接插入实验报告
    with open("comparison_result.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["方法", "平均得分", "平均最大方块", "最佳最大方块", "平均步数"])
        writer.writerow([
            "基础启发式",
            base_result["avg_score"],
            base_result["avg_max_tile"],
            base_result["best_max_tile"],
            base_result["avg_steps"]
        ])
        writer.writerow([
            "进化优化启发式",
            ga_result["avg_score"],
            ga_result["avg_max_tile"],
            ga_result["best_max_tile"],
            ga_result["avg_steps"]
        ])
        writer.writerow([
            "监督学习",
            sl_result["avg_score"],
            sl_result["avg_max_tile"],
            sl_result["best_max_tile"],
            sl_result["avg_steps"]
        ])
        writer.writerow([
            "DQN强化学习",
            dqn_result["avg_score"],
            dqn_result["avg_max_tile"],
            dqn_result["best_max_tile"],
            dqn_result["avg_steps"]
        ])

    print("\n" + "=" * 60)
    print("✅ 所有AI评估完成，对比结果已保存到 comparison_result.csv")