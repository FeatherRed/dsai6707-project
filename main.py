"""Single entry point: run plain DE or LLMDE on the bbob-40D benchmark
(default) or a simple Sphere demo.

Examples:
    python main.py --algorithm de                           # BBOB F1 (Sphere), 40D, frozen instance
    python main.py --algorithm de --function 8 --max-fes 5000 --seed 7
    python main.py --algorithm llmde                        # mock: offline pipeline check only
    python main.py --algorithm llmde --live --request-budget 8
    python main.py --algorithm de --problem sphere --dim 5
"""
import argparse
import json
import math
import time

import torch

from bbob import BBOB_NAMES, BBOBProblem
from de import DE, FixedPolicy, integer, number
from llm import Controller, ControlStop, HTTPTransport, RequestBudget, settings_from_env
from problem import Problem, sphere

GROUPS = 3  # the study fixes K=3 groups for both algorithms


def planned_consultations(args):
    generations = math.ceil((args.max_fes - args.pop_size) / args.pop_size)
    return math.ceil(generations / args.interval)


def validate_args(args):
    if args.problem == "bbob":
        if args.lb is not None or args.ub is not None:
            raise ValueError("--lb/--ub apply to --problem sphere only; BBOB is fixed to [-5,5]^D")
        if args.dtype != "float64":
            raise ValueError("--problem bbob is defined in float64 (frozen benchmark mode)")
        integer(args.function, "--function", 1, 24)
        if args.dim != 40:
            raise ValueError("--problem bbob uses the saved 40D instances; --dim must be 40")
    else:
        if args.function != 1:
            raise ValueError("--function applies to --problem bbob only")
        lb = -5.0 if args.lb is None else args.lb
        ub = 5.0 if args.ub is None else args.ub
        if lb >= ub:
            raise ValueError("--lb must be smaller than --ub")
        integer(args.dim, "--dim", 1, 40)
    integer(args.pop_size, "--pop-size", 6, 50)
    integer(args.max_fes, "--max-fes", args.pop_size, 10**9)
    integer(args.interval, "--interval", 1, 10**6)
    integer(args.seed, "--seed", 0, 2**31 - 5)
    integer(args.retries, "--retries", 0, 3)
    integer(args.max_tokens, "--max-tokens", 1, 100000)
    for key in ("connect_timeout", "read_timeout", "total_timeout"):
        number(getattr(args, key), f"--{key.replace('_', '-')}", 0, 86400, exclusive_low=True)
    number(args.temperature, "--temperature", 0, 2)
    number(args.top_p, "--top-p", 0, 1, exclusive_low=True)
    for key in ("request_interval_seconds", "retry_backoff_seconds", "retry_backoff_max_seconds"):
        number(getattr(args, key), f"--{key.replace('_', '-')}", 0, 3600)
    if args.retry_backoff_seconds > args.retry_backoff_max_seconds:
        raise ValueError("--retry-backoff-seconds must not exceed --retry-backoff-max-seconds")
    if args.reasoning_effort not in (None, "low", "high", "max"):
        raise ValueError("--reasoning-effort must be omitted or low/high/max")
    if args.response_format not in (None, "json_object"):
        raise ValueError("--response-format must be omitted or json_object")
    if args.algorithm == "de":
        number(args.F, "--F", 0, 2, exclusive_low=True)
        number(args.CR, "--CR", 0, 1)
    if args.live:
        if args.algorithm != "llmde":
            raise ValueError("--live applies to --algorithm llmde only")
        integer(args.request_budget, "--request-budget", 1, 10**9)
    elif args.request_budget is not None:
        raise ValueError("--request-budget requires --live")


def build_controller(args):
    if args.algorithm == "de":
        return FixedPolicy(args.pop_size, k=GROUPS, F=args.F, CR=args.CR)
    if not args.live:
        print("LLMDE mock mode: random legal configurations; verifies the pipeline only, "
              "NOT optimization quality. Use --live --request-budget N for real LLM decisions.")
        return Controller("mock", seed=args.seed, k=GROUPS, interval=args.interval)
    settings = settings_from_env()
    budget = RequestBudget(args.request_budget)
    transport = HTTPTransport(live=True, settings=settings, connect_timeout=args.connect_timeout,
                              read_timeout=args.read_timeout, total_timeout=args.total_timeout)
    options = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens,
               "reasoning_effort": args.reasoning_effort, "response_format": args.response_format,
               "request_interval_seconds": args.request_interval_seconds,
               "retry_backoff_seconds": args.retry_backoff_seconds,
               "retry_backoff_max_seconds": args.retry_backoff_max_seconds,
               "verbose_control": args.verbose_control}
    return Controller("llm", seed=args.seed, k=GROUPS, interval=args.interval, transport=transport,
                      budget=budget, settings=settings, options=options, retries=args.retries)


def build_problem(args):
    if args.problem == "bbob":
        return BBOBProblem(args.function)
    lb = -5.0 if args.lb is None else args.lb
    ub = 5.0 if args.ub is None else args.ub
    return Problem(sphere, dim=args.dim, lb=lb, ub=ub, dtype=getattr(torch, args.dtype))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", choices=("de", "llmde"), default="de",
                        help="de: fixed-parameter DE (rand/1, binomial, F/CR fixed); "
                             "llmde: LLM jointly decides grouping, operators and parameters")
    parser.add_argument("--problem", choices=("bbob", "sphere"), default="bbob",
                        help="bbob: frozen bbob-40D benchmark (default); sphere: simple demo function")
    parser.add_argument("--function", type=int, default=1,
                        help="bbob only: BBOB function number 1..24 (default 1)")
    parser.add_argument("--dim", type=int, help="dimension (bbob fixed at 40; sphere defaults to 10)")
    parser.add_argument("--lb", type=float, help="sphere only: native lower bound (default -5)")
    parser.add_argument("--ub", type=float, help="sphere only: native upper bound (default 5)")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64",
                        help="sphere only; bbob is always float64")
    parser.add_argument("--pop-size", type=int, default=50, help="population size N, 6..50 (default 50)")
    parser.add_argument("--max-fes", type=int, default=10000, help="evaluation budget (default 10000)")
    parser.add_argument("--interval", type=int, default=5,
                        help="consultation period H in generations (default 5)")
    parser.add_argument("--seed", type=int, default=2024, help="optimizer seed (default 2024)")
    parser.add_argument("--F", type=float, default=0.5, help="plain DE only: scale factor, 0<F<=2 (default 0.5)")
    parser.add_argument("--CR", type=float, default=0.9, help="plain DE only: crossover rate, 0<=CR<=1 (default 0.9)")
    parser.add_argument("--live", action="store_true", help="llmde only: call the real LLM API")
    parser.add_argument("--request-budget", type=int,
                        help="llmde live only: maximum HTTP attempts (required with --live)")
    parser.add_argument("--retries", type=int, default=0, help="llmde live only: retries per consultation, 0..3")
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=60.0)
    parser.add_argument("--total-timeout", type=float, default=180.0,
                        help="cumulative network wall-clock budget per run in seconds")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"))
    parser.add_argument("--response-format", choices=("json_object",))
    parser.add_argument("--request-interval-seconds", type=float, default=0.0,
                        help="minimum spacing between HTTP attempts")
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--retry-backoff-max-seconds", type=float, default=60.0)
    parser.add_argument("--verbose-control", action="store_true", help="print per-consultation control logs")
    args = parser.parse_args(argv)
    if args.dim is None:
        args.dim = 40 if args.problem == "bbob" else 10
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    mode = {"de": "fixed_de", "llmde": "llmde_live" if args.live else "llmde_mock"}[args.algorithm]
    controller = build_controller(args)
    problem = build_problem(args)
    optimizer = DE(controller, pop_size=args.pop_size, maxfes=args.max_fes, interval=args.interval)
    optimizer.seed(args.seed)

    started = time.perf_counter()
    status, reason = "complete", "fes_budget_completed"
    try:
        optimizer.optimize(problem)
        if args.algorithm == "llmde" and args.live and planned_consultations(args) > 0:
            stats = controller.statistics()
            if stats["valid_configurations"] == 0:
                status, reason = "failed", "no_valid_llm_configuration"
            elif stats["fallbacks"]:
                reason = "fes_budget_completed_with_fallback"
    except ControlStop as exc:
        status, reason = "incomplete", str(exc)
    except KeyboardInterrupt:
        status, reason = "incomplete", "interrupted"
    elapsed = time.perf_counter() - started

    summary = {"algorithm": args.algorithm, "mode": mode, "status": status, "stop_reason": reason,
               "problem": args.problem, "dim": args.dim,
               "pop_size": args.pop_size, "max_fes": args.max_fes,
               "interval": args.interval, "seed": args.seed,
               "actual_fes": optimizer.fes, "generations": optimizer.gen,
               "best_fitness": optimizer.gbest,
               "best_x_native": problem.denormalize(optimizer.pop[int(optimizer.costs.argmin())]).tolist(),
               "planned_consultations": planned_consultations(args) if args.algorithm == "llmde" else 0,
               "compute_seconds": optimizer.compute_seconds,
               "wall_seconds": elapsed}
    if args.problem == "bbob":
        summary.update(function=args.function, function_name=BBOB_NAMES[args.function],
                       instance=problem.instance_id, split=problem.split,
                       optimum_value=problem.optimum,
                       best_error=optimizer.gbest - problem.optimum)
    if args.algorithm == "llmde":
        summary["llm"] = controller.statistics()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "complete" else 130 if reason == "interrupted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
