"""Statistics-only observation of the search and compact configuration feedback.

LLMDE decisions are based exclusively on the features produced here: no raw
decision vectors, function identity, known optimum or landscape parameters
ever leave the optimizer. All features are whitelisted, scalar and compacted.
"""
import copy
import math
from collections import deque

import torch

STATE_VERSION = "population12-individual8-v1"
POPULATION_COLUMNS = ("budget_fraction", "best_gain_total", "best_gain_window", "mean_gap", "fitness_spread", "diversity", "diversity_trend", "success_rate_window", "global_stall_H", "stalled_fraction", "adjacent_corr", "variance_concentration")
INDIVIDUAL_COLUMNS = ("id", "rank", "gap", "gain", "success", "stall_H", "d_best", "d_center", "d_knn", "prev_group")
LAST_GROUP_COLUMNS = ("group", "mutation", "crossover", "F", "CR", "evaluations", "success", "mean_gain")
CREDIT_COLUMNS = ("mutation", "crossover", "F", "CR", "evaluations", "success", "mean_gain")
STATE_KEYS = {"state_version", "N", "D", "fes", "max_fes", "observed_window_generations", "population_columns", "population", "individual_columns", "individuals", "last_group_columns", "last_groups", "credit_columns", "credit"}


def compact(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite statistics feature")
        return float(format(value, ".4g"))
    if isinstance(value, (tuple, list)):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if value is None or type(value) in (str, int):
        return value
    raise ValueError("Unexpected statistics feature type")


def validate_state(state):
    if set(state) != STATE_KEYS or state["state_version"] != STATE_VERSION:
        raise ValueError("Statistics-only input schema mismatch")
    for name in ("N", "D", "fes", "max_fes", "observed_window_generations"):
        if type(state[name]) is not int:
            raise ValueError("Statistics metadata must contain integers only")
    if (state["N"] < 2 or state["D"] < 1 or not state["N"] <= state["fes"] <= state["max_fes"]
            or state["observed_window_generations"] < 0):
        raise ValueError("Statistics metadata outside runtime bounds")
    for name, expected in (("population_columns", POPULATION_COLUMNS), ("individual_columns", INDIVIDUAL_COLUMNS), ("last_group_columns", LAST_GROUP_COLUMNS), ("credit_columns", CREDIT_COLUMNS)):
        if state[name] != list(expected):
            raise ValueError("Unexpected feature columns")
    if len(state["population"]) != 12 or len(state["individuals"]) != state["N"]:
        raise ValueError("Statistics feature shape mismatch")
    for index, row in enumerate(state["individuals"]):
        if len(row) != 10 or type(row[0]) is not int or row[0] != index:
            raise ValueError("Individual rows must cover stable IDs in order")
        if any(value is not None and type(value) not in (int, float) for value in row):
            raise ValueError("Individual rows must contain scalar statistics only")
    if any(value is not None and type(value) not in (int, float) for value in state["population"]):
        raise ValueError("Population vector must contain scalar statistics only")
    if len(state["last_groups"]) > 3 or len(state["credit"]) > 4:
        raise ValueError("Feedback exceeds input budget")
    for rows, columns in ((state["last_groups"], LAST_GROUP_COLUMNS), (state["credit"], CREDIT_COLUMNS)):
        for row in rows:
            if len(row) != len(columns) or any(value is not None and type(value) not in (str, int, float) for value in row):
                raise ValueError("Feedback rows must contain scalar statistics only")
    compact(state)
    return state


def diversity(population):
    # Center differences rather than absolute coordinates so an identical
    # population has exactly zero variance despite floating-point averaging.
    relative = population - population[0]
    return float((relative - relative.mean(0)).square().mean().sqrt())


class StatisticsObservation:
    """Population/individual statistics on the fixed initial fitness scale."""

    def __init__(self, population, fitness, *, interval, max_fes):
        self.interval, self.max_fes = interval, max_fes
        initial = fitness.detach().cpu().double()
        self.unit = max(float(initial.abs().max()), 1.0)
        scaled = initial / self.unit
        # Represent fixed S0 as unit * scale, avoiding IQR overflow.
        self.scale = max(float(torch.quantile(scaled, .75) - torch.quantile(scaled, .25)), 1e-12)
        self.initial_best = float(scaled.min())
        self.begin_window(population, fitness, 0)

    def begin_window(self, population, fitness, generation):
        self.start_fitness = fitness.detach().cpu().double().clone()
        self.start_diversity = diversity(population.detach().cpu().double())
        self.start_generation = generation
        self.evaluations = torch.zeros(len(fitness), dtype=torch.int64)
        self.successes = torch.zeros_like(self.evaluations)

    def record_generation(self, parent, trial, count):
        self.evaluations[:count] += 1
        self.successes[:count] += (trial.detach().cpu() < parent[:count].detach().cpu()).long()

    def relative_gain(self, parent, trial):
        return ((parent.double() / self.unit - trial.double() / self.unit) / self.scale).clamp_min(0)

    def log_ratio(self, scaled_nonnegative):
        value = torch.as_tensor(scaled_nonnegative, dtype=torch.float64).clamp_min(0)
        return torch.logaddexp(torch.zeros_like(value), torch.log(value) - math.log(self.scale))

    def observe(self, pop, costs, *, fes, generation, stagnation, global_stagnation, previous_config):
        x, f = pop.detach().cpu().double(), costs.detach().cpu().double()
        if not torch.isfinite(x).all() or not torch.isfinite(f).all():
            raise ValueError("Nonfinite population or fitness")
        if bool(((x < 0) | (x > 1)).any()):
            raise ValueError("Statistics require normalized runtime coordinates")
        n, dim = x.shape
        scaled = f / self.unit
        best_id = int(torch.argmin(f))
        best = float(scaled[best_id])
        observed = generation - self.start_generation
        relative = x - x[0]
        centered = relative - relative.mean(0)
        var = centered.square().mean(0)
        variance_sum = float(var.sum())
        adjacent = None
        if dim > 1:
            valid = (var[:-1] > 0) & (var[1:] > 0)
            if bool(valid.any()):
                cov = (centered[:, :-1] * centered[:, 1:]).mean(0)
                correlations = cov[valid] / (var[:-1][valid] * var[1:][valid]).sqrt()
                adjacent = float(correlations.abs().clamp(0, 1).mean())
        current_diversity = diversity(x)
        total_evaluations = int(self.evaluations.sum())
        population = [fes / self.max_fes,
                      float(self.log_ratio(max(self.initial_best - best, 0))),
                      float(self.log_ratio(max(float((self.start_fitness / self.unit).min()) - best, 0))) if observed else None,
                      float(self.log_ratio(float(scaled.mean()) - best)),
                      float(self.log_ratio(float(scaled.std(unbiased=False)))), current_diversity,
                      (current_diversity - self.start_diversity) / (current_diversity + self.start_diversity + 1e-12) if observed else None,
                      int(self.successes.sum()) / total_evaluations if total_evaluations else None,
                      min(global_stagnation / self.interval, 5.),
                      float((stagnation >= self.interval).double().mean()) if observed else None,
                      adjacent, float(var.max()) / variance_sum if variance_sum else 0.]
        ranks = (f[:, None] > f[None, :]).sum(1).double() / max(n - 1, 1)
        gap = self.log_ratio(scaled - best)
        gain = self.log_ratio(self.start_fitness / self.unit - scaled)
        distances = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist") / math.sqrt(dim)
        distances.fill_diagonal_(math.inf)
        nearest = distances.topk(min(3, n - 1), largest=False).values.mean(1)
        d_best = (x - x[best_id]).norm(dim=1) / math.sqrt(dim)
        d_center = centered.norm(dim=1) / math.sqrt(dim)
        previous = [-1] * n
        if previous_config is not None:
            for group_id, group in enumerate(previous_config["groups"]):
                for member in group["members"]:
                    previous[member] = group_id
        individuals = [[i, float(ranks[i]), float(gap[i]),
                        float(gain[i]) if int(self.evaluations[i]) else None,
                        int(self.successes[i]) / int(self.evaluations[i]) if int(self.evaluations[i]) else None,
                        min(float(stagnation[i]) / self.interval, 5.),
                        float(d_best[i]), float(d_center[i]), float(nearest[i]), previous[i]] for i in range(n)]
        return {"state_version": STATE_VERSION, "N": n, "D": dim, "fes": fes, "max_fes": self.max_fes,
                "observed_window_generations": observed,
                "population_columns": list(POPULATION_COLUMNS), "population": population,
                "individual_columns": list(INDIVIDUAL_COLUMNS), "individuals": individuals,
                "last_group_columns": list(LAST_GROUP_COLUMNS), "last_groups": [],
                "credit_columns": list(CREDIT_COLUMNS), "credit": []}


def configuration(group):
    return {key: group[key] for key in ("mutation", "crossover", "F", "CR")}


def signature(group):
    return tuple(group[key] for key in ("mutation", "crossover", "F", "CR"))


def metrics(records):
    count = sum(r["evaluations"] for r in records)
    successes = sum(r["successes"] for r in records)
    return {"evaluated_targets": count, "successful_targets": successes,
            "success_rate": successes / count if count else None,
            "mean_gain": math.log1p(sum(r["gain"] for r in records) / count) if count else None}


class ConfigurationFeedback:
    """Compact episode-local credit on the fixed initialization fitness scale."""

    def __init__(self, observer, performance_window=50, capacity=64):
        self.observer = observer
        self.generations = deque(maxlen=performance_window)
        self.experiences = deque(maxlen=capacity)
        self.pending = None

    def start_window(self, state, action, generation):
        self.pending = {"start_population": list(state["population"]), "start_fes": state["fes"],
                        "start_generation": generation, "groups": copy.deepcopy(action["groups"]), "records": []}

    def record_generation(self, parent_costs, trial_costs, action, count):
        gain = self.observer.relative_gain(parent_costs[:count], trial_costs).cpu()
        success = (trial_costs < parent_costs[:count]).cpu()
        records = []
        for index, group in enumerate(action["groups"]):
            ids = [i for i in group["members"] if i < count]
            if ids:
                records.append({"configuration": configuration(group), "group_index": index,
                                "evaluations": len(ids), "successes": int(success[ids].sum()), "gain": float(gain[ids].sum())})
        self.generations.append(records)
        if self.pending is not None:
            self.pending["records"].extend(records)

    def finish_window(self, state, generation):
        if self.pending is None:
            return
        start = self.pending
        groups = [{"group_index": i, "configuration": configuration(group),
                   **metrics([r for r in start["records"] if r["group_index"] == i])}
                  for i, group in enumerate(start["groups"])]
        self.experiences.append({"start_population": start["start_population"], "end_population": state["population"],
                                 "start_fes": start["start_fes"], "end_fes": state["end_fes"],
                                 "generations": generation - start["start_generation"], "groups": groups})
        self.pending = None

    def context(self, state=None):
        last = []
        if self.experiences:
            for group in self.experiences[-1]["groups"]:
                last.append([group["group_index"], *signature(group["configuration"]),
                             group["evaluated_targets"], group["success_rate"], group["mean_gain"]])
        records = [r for generation in self.generations for r in generation]
        recent = list(dict.fromkeys(signature(r["configuration"]) for r in reversed(records)))[:4]
        credit = []
        for key in recent:
            data = metrics([r for r in records if signature(r["configuration"]) == key])
            credit.append([*key, data["evaluated_targets"], data["success_rate"], data["mean_gain"]])
        return compact({"last_groups": last, "credit": credit})
