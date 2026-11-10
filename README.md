# 2048 AI Benchmarks

A course project comparing heuristic search, genetic optimization, behavior cloning, and DQN in the same 2048 environment.

2048多策略课程实验。本人提出项目想法并负责主要非论文技术工作，论文写作另有团队分工；开发采用AI辅助。

## 快速运行

Python 3.12 或 3.13，仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python main.py heuristic --games 2 --seed 2026 --max-steps 300
python main.py console --seed 42
```

训练与完整四策略比较需要PyTorch。仅基础游戏／启发式无需PyTorch；可先安装NumPy运行基础入口。

```bash
python main.py ga --population 30 --generations 80 --games-per-individual 5
python supervised_trainer.py --games 20 --epochs 20 --out-dir output_supervised
python dqn_trainer.py --episodes 50 --max-steps 300 --out-dir output_dqn
python run_comparison.py
```

`best_weights.json` 和既有模型已放在比较脚本预期位置。训练会写入输出目录，请先保留需要的检查点。遗传优化命令输出最优权重与历史，选择替换现有权重时应另存实验配置。

## 方法与文件

- `game.py`、`env2048.py`：游戏逻辑和训练环境。
- `heuristic_ai.py`、`genetic_algorithm.py`：启发式与权重优化。
- `cnn_model.py`、`supervised_trainer.py`：行为克隆。
- `dqn_agent.py`、`dqn_trainer.py`：DQN。
- `run_comparison.py`：统一接口比较。
- `results/historical/`：原交付实验结果与图表，保留历史身份。

## 历史结果

| 方法 | 原CSV中的平均得分 |
| --- | ---: |
| 基础启发式 | 14605.6 |
| 进化优化启发式 | 14960.4 |
| 监督学习 | 6577.6 |
| DQN | 2935.2 |

上述值来自原交付CSV，本次整理未重训或重跑四模型比较。原比较脚本默认10局、种子42、最多1000步；历史CSV缺少绑定配置，不能仅据默认值认定它的实际实验设置。

当前记录中学习策略弱于启发式。后续重点是核对训练预算、非法动作处理与评价协议，加入多种子结果、方差和运行耗时；不将小样本成绩包装成普遍优越性。

本次基础运行检查见 [验证记录](docs/VALIDATION.md)。项目整体许可证尚未指定。
