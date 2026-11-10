"""Command-line entry point for member A's 2048 modules."""

from __future__ import annotations

import argparse

from game import play_console
from genetic_algorithm import GAConfig, GeneticOptimizer
from heuristic_ai import HeuristicAI, evaluate_ai


def run_heuristic(args: argparse.Namespace) -> None:
    ai = HeuristicAI()
    metrics = evaluate_ai(ai, games=args.games, seed=args.seed, max_steps=args.max_steps)
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")


def run_ga(args: argparse.Namespace) -> None:
    config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        games_per_individual=args.games_per_individual,
        workers=args.workers,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    optimizer = GeneticOptimizer(config)
    best_weights, best_fitness, _ = optimizer.run()
    optimizer.save_history(args.history)
    print(f"best_fitness: {best_fitness:.2f}")
    print(f"best_weights: {[round(value, 4) for value in best_weights]}")
    print(f"history_saved_to: {args.history}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2048 AI course project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    console = subparsers.add_parser("console", help="play 2048 manually in the terminal")
    console.add_argument("--seed", type=int, default=None)
    console.set_defaults(func=lambda args: play_console(seed=args.seed))

    heuristic = subparsers.add_parser("heuristic", help="evaluate the heuristic AI")
    heuristic.add_argument("--games", type=int, default=10)
    heuristic.add_argument("--seed", type=int, default=2026)
    heuristic.add_argument("--max-steps", type=int, default=5000)
    heuristic.set_defaults(func=run_heuristic)

    ga = subparsers.add_parser("ga", help="run genetic optimization")
    ga.add_argument("--population", type=int, default=30)
    ga.add_argument("--generations", type=int, default=80)
    ga.add_argument("--games-per-individual", type=int, default=5)
    ga.add_argument("--workers", type=int, default=4)
    ga.add_argument("--seed", type=int, default=2026)
    ga.add_argument("--max-steps", type=int, default=5000)
    ga.add_argument("--history", default="ga_history.json")
    ga.set_defaults(func=run_ga)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
