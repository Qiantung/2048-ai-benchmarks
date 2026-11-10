# 验证记录

2026-11-10，Windows，Python 3.12.14，NumPy 2.3.5。

执行 `python main.py heuristic --games 2 --seed 2026 --max-steps 300` 成功，平均得分4362、最佳得分4576、平均最大方块384、最佳最大方块512；两局均达到300步上限。

这是基础入口的短运行检查，不是完整策略基准。未重训CNN或DQN，未加载其检查点运行完整比较；`results/historical/` 保留的成绩和图表仍是历史记录。
