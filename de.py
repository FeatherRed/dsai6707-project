"""Shared DE kernel: operators, configuration contract and the optimizer loop.

Plain DE and LLMDE run the exact same synchronous DE kernel over normalized
[0,1] coordinates. They differ only in the controller: every ``interval``
generations the controller decides how the population is grouped and which
joint (mutation, crossover, F, CR) configuration each group uses. A
:class:`FixedPolicy` yields standard fixed-parameter DE; an LLM controller
(see llm.py) yields LLMDE.
"""
import json
import math
import time

import torch

from state import ConfigurationFeedback, StatisticsObservation, compact, validate_state

MUTATIONS = ("rand/1", "rand/2", "best/1", "best/2", "current-to-best/1")
CROSSOVERS = ("binomial", "exponential")


def number(value, name, low, high, exclusive_low=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number (not bool)")
    if value > high or value < low or (exclusive_low and value == low):
        raise ValueError(f"{name} outside allowed range")
    return value


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def validate_groups(payload, n, k=3):
    """Strict groups contract: K nonempty groups covering IDs 0..N-1 exactly once."""
    if not isinstance(payload, dict) or set(payload) != {"groups"}:
        raise ValueError("Expected exactly the 'groups' field")
    groups = payload["groups"]
    if not isinstance(groups, list) or len(groups) != k:
        raise ValueError(f"Expected exactly {k} groups")
    seen = []
    for group in groups:
        if not isinstance(group, dict) or set(group) != {"members", "mutation", "crossover", "F", "CR"}:
            raise ValueError("Unexpected or missing group fields")
        ids = group["members"]
        if not isinstance(ids, list) or not ids:
            raise ValueError("Each group must have nonempty members")
        for slot in ids:
            integer(slot, "member ID", 0, n - 1)
        seen.extend(ids)
        if group["mutation"] not in MUTATIONS or group["crossover"] not in CROSSOVERS:
            raise ValueError("Unknown mutation or crossover")
        number(group["F"], "F", 0, 2, exclusive_low=True)
        number(group["CR"], "CR", 0, 1)
    if sorted(seen) != list(range(n)):
        missing = sorted(set(range(n)) - set(seen))
        duplicates = sorted({i for i in seen if seen.count(i) > 1})
        raise ValueError(f"IDs must cover 0..N-1 exactly once; missing={missing}; duplicates={duplicates}")
    # Return a detached JSON value, never a mutable caller-owned object.
    return json.loads(json.dumps(payload, allow_nan=False))


def fallback_groups(n, k=3, F=0.5, CR=0.5):
    """Deterministic ID cohorts with one shared configuration.

    This is the standard fixed-DE configuration and the LLM failure fallback;
    it is never labelled LLM grouping.
    """
    return {"groups": [{"members": list(range(i * n // k, (i + 1) * n // k)),
                        "mutation": "rand/1", "crossover": "binomial", "F": F, "CR": CR}
                       for i in range(k)]}


class FixedPolicy:
    """Plain DE controller: fixed configuration (default rand/1/bin, F=CR=0.5)."""

    def __init__(self, n, k=3, F=0.5, CR=0.5):
        self.action = validate_groups(fallback_groups(n, k, F, CR), n, k)

    def consult(self, state):
        return self.action


def sample_donors(n, target, best, mutation, rng, device="cpu"):
    count = {"rand/1": 3, "rand/2": 5, "best/1": 2, "best/2": 4, "current-to-best/1": 2}[mutation]
    # All random indices are mutually distinct, exclude target; best-based
    # operators also exclude best. Target==best remains a legal target.
    excluded = {target}
    if mutation in ("best/1", "best/2", "current-to-best/1"):
        excluded.add(best)
    candidates = [i for i in range(n) if i not in excluded]
    if len(candidates) < count:
        raise ValueError("Insufficient legal donors")
    order = torch.randperm(len(candidates), generator=rng, device=device).cpu().tolist()
    return [candidates[i] for i in order[:count]]


def mutate(pop, target, best, mutation, F, donors):
    r = [pop[i] for i in donors]
    if mutation == "rand/1":
        return r[0] + F * (r[1] - r[2])
    if mutation == "rand/2":
        return r[0] + F * (r[1] - r[2]) + F * (r[3] - r[4])
    if mutation == "best/1":
        return pop[best] + F * (r[0] - r[1])
    if mutation == "best/2":
        return pop[best] + F * (r[0] - r[1]) + F * (r[2] - r[3])
    if mutation == "current-to-best/1":
        return pop[target] + F * (pop[best] - pop[target]) + F * (r[0] - r[1])
    raise ValueError("Unknown mutation")


def repair(v, rng):
    bad = (v < 0) | (v > 1) | ~torch.isfinite(v)
    out = v.clone()
    out[bad] = torch.rand((int(bad.sum()),), device=v.device, dtype=v.dtype, generator=rng)
    return out


def crossover(target, mutant, kind, CR, rng):
    dim = target.numel()
    start = int(torch.randint(dim, (1,), generator=rng, device=target.device))
    mask = torch.zeros(dim, dtype=torch.bool, device=target.device)
    if kind == "binomial":
        mask = torch.rand(dim, generator=rng, device=target.device) < CR
        mask[start] = True
    elif kind == "exponential":
        mask[start] = True
        length = 1
        while length < dim and float(torch.rand((), generator=rng, device=target.device)) < CR:
            mask[(start + length) % dim] = True
            length += 1
    else:
        raise ValueError("Unknown crossover")
    return torch.where(mask, mutant, target)


class DE:
    """Synchronous DE over normalized [0,1] coordinates with exact FEs.

    RNG streams: initialization uses ``seed``, variation/repair uses ``seed+1``.
    Donors come from the whole parent population; one generation is produced
    from a parent snapshot, and only strict improvements replace their parent.
    Initialization counts towards the budget; a final partial generation
    evaluates only the low-ID remaining targets.
    """

    def __init__(self, controller, pop_size=50, maxfes=1550, interval=10):
        if pop_size < 6 or maxfes < pop_size:
            raise ValueError("Require N>=6 and max_fes>=N")
        self.controller, self.pop_size, self.maxfes, self.interval = controller, pop_size, maxfes, interval

    def seed(self, seed=None, device="cpu"):
        rng_seed = int(time.time()) if seed is None else seed
        self.rng_seed, self.device = rng_seed, device
        self.init_rng = torch.Generator(device).manual_seed(rng_seed)
        self.rng = torch.Generator(device).manual_seed(rng_seed + 1)

    def init_population(self, problem):
        self.pop = torch.rand((self.pop_size, problem.dim), generator=self.init_rng, device=self.device,
                              dtype=getattr(problem, "dtype", torch.float32))
        self.costs = self._evaluate(problem, self.pop)
        self.fes, self.gen = self.pop_size, 0
        self.gbest = float(self.costs.min())
        self.fes_list, self.gbest_list = [self.fes], [self.gbest]
        self.costs_list = [self.costs.cpu().numpy().copy()]
        self.stagnation = torch.zeros(self.pop_size, dtype=torch.int64)
        self.global_stagnation = 0
        self.active = None
        self.observer = StatisticsObservation(self.pop, self.costs, interval=self.interval, max_fes=self.maxfes)
        self.feedback = ConfigurationFeedback(self.observer)

    @staticmethod
    def _evaluate(problem, x):
        result = problem.eval(x).reshape(-1)
        if result.numel() != len(x) or not torch.isfinite(result).all():
            raise ValueError("Objective must return one finite value per individual")
        return result

    def observe(self):
        state = self.observer.observe(self.pop, self.costs, fes=self.fes, generation=self.gen,
                                      stagnation=self.stagnation, global_stagnation=self.global_stagnation,
                                      previous_config=self.active)
        state.update(self.feedback.context())
        return validate_state(compact(state))

    def optimize(self, problem):
        self.compute_seconds = 0.0
        started = time.perf_counter()
        self.init_population(problem)
        self.compute_seconds += time.perf_counter() - started
        while self.fes < self.maxfes:
            if self.gen % self.interval == 0:
                started = time.perf_counter()
                state = self.observe()
                self.feedback.finish_window(state, self.gen)
                state = self.observe()
                self.compute_seconds += time.perf_counter() - started
                self.active = self.controller.consult(state)
                self.feedback.start_window(state, self.active, self.gen)
                self.observer.begin_window(self.pop, self.costs, self.gen)
            started = time.perf_counter()
            try:
                self.update(self.active, problem)
            finally:
                self.compute_seconds += time.perf_counter() - started
        self.feedback.finish_window(self.observe(), self.gen)
        return self.get_results()

    def update(self, action, problem):
        # No in-place parent mutation until ALL trials have been generated/evaluated.
        parent = self.pop.clone()
        best_id = int(torch.argmin(self.costs))
        configs = {i: group for group in action["groups"] for i in group["members"]}
        count = min(self.pop_size, self.maxfes - self.fes)
        if count == 0:
            return self.observe(), True
        trials = []
        for target in range(count):
            cfg = configs[target]
            donors = sample_donors(self.pop_size, target, best_id, cfg["mutation"], self.rng, self.device)
            mutant = repair(mutate(parent, target, best_id, cfg["mutation"], cfg["F"], donors), self.rng)
            trials.append(crossover(parent[target], mutant, cfg["crossover"], cfg["CR"], self.rng))
        trial = torch.stack(trials)
        costs = self._evaluate(problem, trial)
        self.observer.record_generation(self.costs, costs, count)
        self.feedback.record_generation(self.costs, costs, action, count)
        self.fes += count
        improved = costs < self.costs[:count]
        self.stagnation[:count] += 1
        self.stagnation[:count][improved.cpu()] = 0
        ids = torch.where(improved)[0]
        self.pop[ids] = trial[ids]
        self.costs[ids] = costs[ids]
        previous_best = self.gbest
        self.gbest = float(self.costs.min())
        self.global_stagnation = self.global_stagnation + 1 if self.gbest >= previous_best else 0
        self.gen += 1
        self.fes_list.append(self.fes)
        self.gbest_list.append(self.gbest)
        self.costs_list.append(self.costs.cpu().numpy().copy())
        return self.observe(), self.fes >= self.maxfes

    def get_results(self):
        return {"pop": self.pop.cpu(), "costs": self.costs.cpu(), "fes_list": self.fes_list,
                "gbest_list": self.gbest_list, "costs_list": self.costs_list, "gen": self.gen}
