"""Genetic algorithm optimizer for the 2048 heuristic weights."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import random
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from heuristic_ai import HeuristicAI, HeuristicWeights


Individual = List[float]


@dataclass
class GAConfig:
    population_size: int = 30
    generations: int = 80
    games_per_individual: int = 5
    tournament_size: int = 3
    elite_count: int = 2
    mutation_rate: float = 0.10
    mutation_strength: float = 0.20
    weight_min: float = -5.0
    weight_max: float = 5.0
    workers: int = 4
    seed: int = 2026
    max_steps: int = 5000


@dataclass
class GenerationRecord:
    generation: int
    best_fitness: float
    average_fitness: float
    best_weights: Individual


@dataclass
class GeneticOptimizer:
    config: GAConfig = field(default_factory=GAConfig)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.weight_count = len(HeuristicWeights().as_list())
        self.history: List[GenerationRecord] = []

    def random_individual(self) -> Individual:
        return [
            self.rng.uniform(self.config.weight_min, self.config.weight_max)
            for _ in range(self.weight_count)
        ]

    def initialize_population(self) -> List[Individual]:
        population = [self.random_individual() for _ in range(self.config.population_size - 1)]
        population.append(HeuristicWeights().as_list())
        return population

    def fitness(self, individual: Sequence[float], seed_offset: int = 0) -> float:
        ai = HeuristicAI(individual)
        seed = self.config.seed + seed_offset
        scores = [
            ai.play_game(seed=seed + game_index, max_steps=self.config.max_steps)["score"]
            for game_index in range(self.config.games_per_individual)
        ]
        return mean(scores)

    def evaluate_population(self, population: List[Individual]) -> List[Tuple[Individual, float]]:
        if self.config.workers <= 1:
            return [(individual, self.fitness(individual, i * 1000)) for i, individual in enumerate(population)]

        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            futures = [
                executor.submit(self.fitness, individual, index * 1000)
                for index, individual in enumerate(population)
            ]
            return [(individual, future.result()) for individual, future in zip(population, futures)]

    def tournament_select(self, scored_population: List[Tuple[Individual, float]]) -> Individual:
        contenders = self.rng.sample(scored_population, self.config.tournament_size)
        return max(contenders, key=lambda item: item[1])[0][:]

    def crossover(self, parent_a: Individual, parent_b: Individual) -> Individual:
        return [
            weight_a if self.rng.random() < 0.5 else weight_b
            for weight_a, weight_b in zip(parent_a, parent_b)
        ]

    def mutate(self, individual: Individual) -> Individual:
        mutated = individual[:]
        for index, value in enumerate(mutated):
            if self.rng.random() < self.config.mutation_rate:
                value += self.rng.uniform(-self.config.mutation_strength, self.config.mutation_strength)
                mutated[index] = clamp(value, self.config.weight_min, self.config.weight_max)
        return mutated

    def next_generation(self, scored_population: List[Tuple[Individual, float]]) -> List[Individual]:
        ranked = sorted(scored_population, key=lambda item: item[1], reverse=True)
        next_population = [individual[:] for individual, _ in ranked[: self.config.elite_count]]

        while len(next_population) < self.config.population_size:
            parent_a = self.tournament_select(scored_population)
            parent_b = self.tournament_select(scored_population)
            child = self.mutate(self.crossover(parent_a, parent_b))
            next_population.append(child)

        return next_population

    def run(self) -> Tuple[Individual, float, List[GenerationRecord]]:
        population = self.initialize_population()
        best_individual: Optional[Individual] = None
        best_fitness = float("-inf")

        for generation in range(1, self.config.generations + 1):
            scored = self.evaluate_population(population)
            generation_best, generation_best_fitness = max(scored, key=lambda item: item[1])
            average_fitness = mean(fitness for _, fitness in scored)

            if generation_best_fitness > best_fitness:
                best_individual = generation_best[:]
                best_fitness = generation_best_fitness

            record = GenerationRecord(
                generation=generation,
                best_fitness=generation_best_fitness,
                average_fitness=average_fitness,
                best_weights=generation_best[:],
            )
            self.history.append(record)
            print(
                f"generation={generation:03d} "
                f"best={generation_best_fitness:.2f} "
                f"avg={average_fitness:.2f} "
                f"weights={[round(value, 3) for value in generation_best]}"
            )

            population = self.next_generation(scored)

        assert best_individual is not None
        return best_individual, best_fitness, self.history

    def save_history(self, path: str) -> None:
        payload = {
            "config": self.config.__dict__,
            "history": [record.__dict__ for record in self.history],
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def main() -> None:
    demo_config = GAConfig(
        population_size=8,
        generations=3,
        games_per_individual=2,
        workers=4,
        max_steps=1500,
    )
    optimizer = GeneticOptimizer(demo_config)
    best_weights, best_fitness, _ = optimizer.run()
    optimizer.save_history("ga_history_demo.json")
    print(f"best_fitness={best_fitness:.2f}")
    print(f"best_weights={[round(value, 4) for value in best_weights]}")


if __name__ == "__main__":
    main()
