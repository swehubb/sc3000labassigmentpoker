# main_part2.py
# SC3000 / CZ3005 Lab Assignment 1 - Part 2 (Grid World MDP/RL)
# Task 1: Value Iteration + Policy Iteration (known model, deterministic, gamma=0.9)
# Task 2: Monte Carlo control (unknown model, stochastic sampling, eps-greedy, eps=0.1)
# Task 3: Tabular Q-learning (unknown model, stochastic sampling, eps-greedy, eps=0.1, alpha=0.1)

import argparse
import random
import math
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# ====== Environment spec (hard-coded from assignment) ======
N = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = {(2, 1), (2, 3)}
ACTIONS = ["U", "D", "L", "R"]
GAMMA = 0.9

# Stochastic transition probabilities (used for Task 2/3 environment interaction)
P_INTENDED = 0.8
P_LEFT = 0.1
P_RIGHT = 0.1

# RL hyperparameters (as assignment)
EPSILON = 0.1
ALPHA = 0.1

# Safety cap for sampled episodes in Task 2/3
MAX_STEPS_PER_EPISODE = 500


# ====== Basic environment helpers ======
def in_bounds(s: Tuple[int, int]) -> bool:
    x, y = s
    return 0 <= x < N and 0 <= y < N


def is_terminal(s: Tuple[int, int]) -> bool:
    return s == GOAL


def is_blocked(s: Tuple[int, int]) -> bool:
    return s in OBSTACLES


def all_states() -> List[Tuple[int, int]]:
    states = []
    for x in range(N):
        for y in range(N):
            s = (x, y)
            if is_blocked(s):
                continue
            states.append(s)
    return states


def move_once(s: Tuple[int, int], a: str) -> Tuple[int, int]:
    """
    Return the next state after applying action a once.
    If the move goes out of grid or into a roadblock, the agent stays in place.
    """
    if is_terminal(s):
        return s

    x, y = s
    if a == "U":
        ns = (x, y + 1)
    elif a == "D":
        ns = (x, y - 1)
    elif a == "L":
        ns = (x - 1, y)
    else:  # "R"
        ns = (x + 1, y)

    if (not in_bounds(ns)) or is_blocked(ns):
        return s
    return ns


def step_deterministic(s: Tuple[int, int], a: str) -> Tuple[Tuple[int, int], float, bool]:
    """
    Deterministic transition used by Task 1 planning.
    Also used as the base movement primitive for stochastic sampling in Task 2/3.
    """
    if is_terminal(s):
        return s, 0.0, True

    ns = move_once(s, a)
    if ns == GOAL:
        return ns, 10.0, True
    return ns, -1.0, False


def perpendicular_actions(a: str) -> Tuple[str, str]:
    """
    Two perpendicular actions used for stochastic slips.
    """
    if a in ("U", "D"):
        return ("L", "R")
    return ("U", "D")


# ====== Task 1 model: deterministic transition model ======
def deterministic_transition_model(
    s: Tuple[int, int], a: str
) -> List[Tuple[float, Tuple[int, int], float, bool]]:
    """
    Known model for Task 1:
    deterministic transitions only, as required by the assignment.
    Returns a list with exactly one outcome: (1.0, next_state, reward, done)
    """
    ns, r, done = step_deterministic(s, a)
    return [(1.0, ns, r, done)]


# ====== Task 2/3 environment: stochastic sampling ======
def sample_step(s: Tuple[int, int], a: str) -> Tuple[Tuple[int, int], float, bool]:
    """
    Unknown-model interaction for Task 2/3:
    intended action with 0.8, perpendicular left/right with 0.1 each.
    """
    if is_terminal(s):
        return s, 0.0, True

    left_a, right_a = perpendicular_actions(a)
    rnd = random.random()
    if rnd < P_INTENDED:
        actual_a = a
    elif rnd < P_INTENDED + P_LEFT:
        actual_a = left_a
    else:
        actual_a = right_a

    return step_deterministic(s, actual_a)


# ====== Printing helpers ======
def policy_arrow(a: Optional[str]) -> str:
    if a is None:
        return "·"
    return {"U": "↑", "D": "↓", "L": "←", "R": "→"}[a]


def print_grid_values(V: Dict[Tuple[int, int], float], title: str) -> None:
    print(title)
    for y in reversed(range(N)):
        row = []
        for x in range(N):
            s = (x, y)
            if s in OBSTACLES:
                row.append("   X    ")
            elif s == GOAL:
                row.append("   G    ")
            else:
                row.append(f"{V.get(s, 0.0):7.2f}")
        print(" ".join(row))
    print()


def print_grid_policy(pi: Dict[Tuple[int, int], str], title: str) -> None:
    print(title)
    for y in reversed(range(N)):
        row = []
        for x in range(N):
            s = (x, y)
            if s in OBSTACLES:
                row.append("X")
            elif s == GOAL:
                row.append("G")
            else:
                row.append(policy_arrow(pi.get(s)))
        print(" ".join(f"{c:>2}" for c in row))
    print()


def compare_policies(
    pi1: Dict[Tuple[int, int], str],
    pi2: Dict[Tuple[int, int], str],
    name1: str,
    name2: str,
) -> None:
    diffs = []
    for s in all_states():
        if is_terminal(s):
            continue
        a1 = pi1.get(s)
        a2 = pi2.get(s)
        if a1 != a2:
            diffs.append((s, a1, a2))

    if not diffs:
        print(f"{name1} and {name2} policies are identical on all non-terminal states.")
    else:
        print(f"{name1} and {name2} policies differ at {len(diffs)} state(s):")
        for s, a1, a2 in diffs:
            print(f"  State {s}: {name1}={a1}, {name2}={a2}")
    print()


# ====== Task 1: Value Iteration & Policy Iteration (deterministic model) ======
def value_iteration(
    theta: float = 1e-6, max_iters: int = 10000
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
    V = {s: 0.0 for s in all_states()}
    V[GOAL] = 0.0

    for _ in range(max_iters):
        delta = 0.0
        for s in all_states():
            if is_terminal(s):
                continue

            old_v = V[s]
            best_q = -math.inf

            for a in ACTIONS:
                q = 0.0
                for p, ns, r, done in deterministic_transition_model(s, a):
                    q += p * (r + (0.0 if done else GAMMA * V[ns]))
                best_q = max(best_q, q)

            V[s] = best_q
            delta = max(delta, abs(old_v - V[s]))

        if delta < theta:
            break

    pi: Dict[Tuple[int, int], str] = {}
    for s in all_states():
        if is_terminal(s):
            continue

        best_a = ACTIONS[0]
        best_q = -math.inf
        for a in ACTIONS:
            q = 0.0
            for p, ns, r, done in deterministic_transition_model(s, a):
                q += p * (r + (0.0 if done else GAMMA * V[ns]))
            if q > best_q:
                best_q = q
                best_a = a
        pi[s] = best_a

    return V, pi


def policy_evaluation(
    pi: Dict[Tuple[int, int], str],
    theta: float = 1e-6,
    max_iters: int = 10000,
) -> Dict[Tuple[int, int], float]:
    V = {s: 0.0 for s in all_states()}
    V[GOAL] = 0.0

    for _ in range(max_iters):
        delta = 0.0
        for s in all_states():
            if is_terminal(s):
                continue

            old_v = V[s]
            a = pi[s]

            v = 0.0
            for p, ns, r, done in deterministic_transition_model(s, a):
                v += p * (r + (0.0 if done else GAMMA * V[ns]))

            V[s] = v
            delta = max(delta, abs(old_v - V[s]))

        if delta < theta:
            break

    return V


def policy_iteration() -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
    pi = {s: "U" for s in all_states() if not is_terminal(s)}

    while True:
        V = policy_evaluation(pi)
        stable = True

        for s in all_states():
            if is_terminal(s):
                continue

            old_a = pi[s]
            best_a = old_a
            best_q = -math.inf

            for a in ACTIONS:
                q = 0.0
                for p, ns, r, done in deterministic_transition_model(s, a):
                    q += p * (r + (0.0 if done else GAMMA * V[ns]))
                if q > best_q:
                    best_q = q
                    best_a = a

            pi[s] = best_a
            if best_a != old_a:
                stable = False

        if stable:
            return V, pi


# ====== RL helpers ======
def epsilon_greedy_action(
    Q: Dict[Tuple[Tuple[int, int], str], float],
    s: Tuple[int, int],
    epsilon: float,
) -> str:
    if random.random() < epsilon:
        return random.choice(ACTIONS)

    best_a = ACTIONS[0]
    best_q = -math.inf
    for a in ACTIONS:
        q = Q.get((s, a), 0.0)
        if q > best_q:
            best_q = q
            best_a = a
    return best_a


def greedy_policy_from_Q(
    Q: Dict[Tuple[Tuple[int, int], str], float]
) -> Dict[Tuple[int, int], str]:
    pi: Dict[Tuple[int, int], str] = {}
    for s in all_states():
        if is_terminal(s):
            continue

        best_a = ACTIONS[0]
        best_q = -math.inf
        for a in ACTIONS:
            q = Q.get((s, a), 0.0)
            if q > best_q:
                best_q = q
                best_a = a
        pi[s] = best_a

    return pi


def V_from_Q(Q: Dict[Tuple[Tuple[int, int], str], float]) -> Dict[Tuple[int, int], float]:
    V: Dict[Tuple[int, int], float] = {}
    for s in all_states():
        if is_terminal(s):
            V[s] = 0.0
        else:
            V[s] = max(Q.get((s, a), 0.0) for a in ACTIONS)
    return V


# ====== Task 2: Monte Carlo Control ======
def monte_carlo_control(
    episodes: int,
    epsilon: float = EPSILON,
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
) -> Tuple[Dict[Tuple[Tuple[int, int], str], float], Dict[Tuple[int, int], str]]:
    Q: Dict[Tuple[Tuple[int, int], str], float] = {}
    returns_sum: Dict[Tuple[Tuple[int, int], str], float] = {}
    returns_cnt: Dict[Tuple[Tuple[int, int], str], int] = {}

    for _ in range(episodes):
        s = START
        episode: List[Tuple[Tuple[int, int], str, float]] = []

        for _step in range(max_steps_per_episode):
            if is_terminal(s):
                break

            a = epsilon_greedy_action(Q, s, epsilon)
            ns, r, done = sample_step(s, a)
            episode.append((s, a, r))
            s = ns

            if done:
                break

        # First-visit MC update
        G = 0.0
        seen = set()
        for s_t, a_t, r_t in reversed(episode):
            G = r_t + GAMMA * G
            if (s_t, a_t) in seen:
                continue
            seen.add((s_t, a_t))

            returns_sum[(s_t, a_t)] = returns_sum.get((s_t, a_t), 0.0) + G
            returns_cnt[(s_t, a_t)] = returns_cnt.get((s_t, a_t), 0) + 1
            Q[(s_t, a_t)] = returns_sum[(s_t, a_t)] / returns_cnt[(s_t, a_t)]

    pi = greedy_policy_from_Q(Q)
    return Q, pi


# ====== Task 3: Q-learning ======
def q_learning(
    episodes: int,
    epsilon: float = EPSILON,
    alpha: float = ALPHA,
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
) -> Tuple[Dict[Tuple[Tuple[int, int], str], float], Dict[Tuple[int, int], str]]:
    Q: Dict[Tuple[Tuple[int, int], str], float] = {}

    for _ in range(episodes):
        s = START

        for _step in range(max_steps_per_episode):
            if is_terminal(s):
                break

            a = epsilon_greedy_action(Q, s, epsilon)
            ns, r, done = sample_step(s, a)

            old_q = Q.get((s, a), 0.0)
            best_next = 0.0 if done else max(Q.get((ns, na), 0.0) for na in ACTIONS)
            target = r + GAMMA * best_next
            Q[(s, a)] = old_q + alpha * (target - old_q)

            s = ns
            if done:
                break

    pi = greedy_policy_from_Q(Q)
    return Q, pi


# ====== Argument handling (works for both CLI and VS Code launch.json) ======
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Part 2 GridWorld MDP/RL. Run one task at a time.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Task number: 1=VI+PI, 2=Monte Carlo Control, 3=Q-learning",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20000,
        help="Episodes for Task 2/3 (default: 20000)",
    )
    return parser


def try_load_launch_args() -> Optional[List[str]]:
    """
    If you click 'Run' with no CLI args, try to read args from launch.json
    in the same folder as this file.
    """
    base = Path(__file__).resolve().parent
    launch_path = base / "launch.json"
    if not launch_path.exists():
        return None

    try:
        data = json.loads(launch_path.read_text(encoding="utf-8"))
        configs = data.get("configurations", [])
        if not configs:
            return None

        # Prefer Task 1 config if it exists; otherwise use the first config.
        chosen = None
        for c in configs:
            name = str(c.get("name", ""))
            if "Task 1" in name or "task 1" in name:
                chosen = c
                break
        if chosen is None:
            chosen = configs[0]

        args = chosen.get("args")
        if isinstance(args, list) and all(isinstance(x, str) for x in args):
            return args
        return None
    except Exception:
        return None


# ====== Main ======
def main():
    random.seed(0)

    parser = build_parser()

    # Case 1: click Run / no CLI args -> try launch.json -> else default Task 1
    if len(sys.argv) == 1:
        launch_args = try_load_launch_args()
        if launch_args is not None:
            args = parser.parse_args(launch_args)
        else:
            args = parser.parse_args(["--task", "1"])
    else:
        # Case 2: CLI args provided directly
        args = parser.parse_args()

    if args.task is None:
        args.task = 1

    if args.task == 1:
        V_vi, pi_vi = value_iteration()
        V_pi, pi_pi = policy_iteration()

        print_grid_values(V_vi, "Value Iteration: V(s)")
        print_grid_policy(pi_vi, "Value Iteration: policy")

        print_grid_values(V_pi, "Policy Iteration: V(s)")
        print_grid_policy(pi_pi, "Policy Iteration: policy")

        compare_policies(pi_vi, pi_pi, "Value Iteration", "Policy Iteration")

    elif args.task == 2:
        Q_mc, pi_mc = monte_carlo_control(episodes=args.episodes, epsilon=EPSILON)
        V_mc = V_from_Q(Q_mc)

        # Compare with optimal policy from Task 1
        _, pi_opt = value_iteration()

        print_grid_values(V_mc, f"Monte Carlo Control (eps={EPSILON}): V(s)")
        print_grid_policy(pi_mc, "Monte Carlo Control: policy")
        compare_policies(pi_mc, pi_opt, "Monte Carlo", "Optimal (VI)")

    else:
        Q_ql, pi_ql = q_learning(episodes=args.episodes, epsilon=EPSILON, alpha=ALPHA)
        V_ql = V_from_Q(Q_ql)

        # Compare with MC and optimal policy
        Q_mc, pi_mc = monte_carlo_control(episodes=args.episodes, epsilon=EPSILON)
        _, pi_opt = value_iteration()

        print_grid_values(V_ql, f"Q-learning (eps={EPSILON}, alpha={ALPHA}): V(s)")
        print_grid_policy(pi_ql, "Q-learning: policy")
        compare_policies(pi_ql, pi_mc, "Q-learning", "Monte Carlo")
        compare_policies(pi_ql, pi_opt, "Q-learning", "Optimal (VI)")


if __name__ == "__main__":
    main()