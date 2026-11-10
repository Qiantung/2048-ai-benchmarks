# 2048 AI Benchmarks

我在这项课程实验中比较了四种玩 2048 的方法：手工启发式、遗传算法优化启发式权重、模仿启发式决策的行为克隆，以及通过奖励学习的 DQN。

项目想法由我提出，主要技术实现也由我负责，论文写作另有团队分工。游戏环境保持一致，便于观察不同决策方法的表现。已有实验记录中，启发式的得分高于两个学习模型。

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

`best_weights.json` 已放在比较脚本预期位置。模型权重没有上传，运行四策略比较前需要先训练对应模型。比较脚本默认读取 `output_dqn_4k/dqn_policy.pt`，DQN训练时请将 `--out-dir` 设为 `output_dqn_4k`；模型路径和训练预算应随实验记录。训练会写入输出目录，请先保留需要的检查点。遗传优化命令输出最优权重与历史，选择替换现有权重时应另存实验配置。

## 方法与文件

- `game.py`、`env2048.py`：游戏逻辑和训练环境。
- `heuristic_ai.py`、`genetic_algorithm.py`：启发式与权重优化。
- `cnn_model.py`、`supervised_trainer.py`：行为克隆。
- `dqn_agent.py`、`dqn_trainer.py`：DQN。
- `run_comparison.py`：统一接口比较。
- `results/historical/`：课程实验留下的结果与图表。

## 已有实验结果

| 方法 | 原CSV中的平均得分 |
| --- | ---: |
| 基础启发式 | 14605.6 |
| 进化优化启发式 | 14960.4 |
| 监督学习 | 6577.6 |
| DQN | 2935.2 |

这些数值来自课程实验保存的 CSV，还没有重新训练并完成新一轮四策略比较。CSV 没有绑定完整实验配置；比较脚本目前的默认设置是 10 局、种子 42、每局最多 1000 步，不能直接视为这组旧结果的设置。

这组结果留下了一个需要继续解释的问题：学习模型为什么没有超过启发式？我下一步要核对训练预算、非法动作处理和评价设置，并补充多个随机种子下的波动与运行耗时。现有记录还不足以说明哪一种方法普遍更好。

开发和代码整理中使用了 AI 辅助。基础入口的运行情况见 [运行记录](docs/VALIDATION.md)。项目整体许可证尚未指定。
