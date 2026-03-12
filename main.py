#!/usr/bin/env python3
"""SC3000/CZ3005 Lab Assignment 1

Part 1: Constrained shortest path with energy budget (NYC instance)
Part 2: Grid-world MDP/RL (value iteration, policy iteration, MC, Q-learning)
"""

from __future__ import annotations

import argparse
import ast
import heapq
import json
import math
import os
import pickle
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


# ------------------------------
# Utilities
# ------------------------------


def load_dict(path: str) -> dict:
    """Load a dictionary file from .pkl/.pickle/.json/.py/.txt.

    - .pkl/.pickle: pickle
    - .json: JSON
    - .py/.txt: literal dict (safe eval)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            return pickle.load(f)
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if ext in {".py", ".txt"}:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return ast.literal_eval(content)
    # fallback: try pickle then literal
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return ast.literal_eval(content)


def find_instance_files(base_dir: str) -> Optional[Tuple[str, str, str, str]]:
    """Try to locate G, Coord, Dist, Cost dictionary files in base_dir."""
    files = os.listdir(base_dir)
    def pick(keyword: str) -> Optional[str]:
        candidates = [f for f in files if keyword.lower() in f.lower()]
        if not candidates:
            return None
        # prefer pickle
        candidates.sort(key=lambda x: (0 if os.path.splitext(x)[1].lower() in {".pkl", ".pickle"} else 1, x))
        return os.path.join(base_dir, candidates[0])

    g = pick("g")
    coord = pick("coord")
    dist = pick("dist")
    cost = pick("cost")
    if all([g, coord, dist, cost]):
        return g, coord, dist, cost
    return None


# ------------------------------
# Part 1: Constrained shortest path
# ------------------------------


@dataclass(order=True)
class Label:
    f: float
    dist: float
    energy: float
    node: str
    prev: Optional[Tuple[str, float, float]]


def edge_key(u: str, v: str) -> str:
    return f"{u},{v}"


def edge_value(d: dict, u: str, v: str, default: float = float("inf")) -> float:
    key = edge_key(u, v)
    if key in d:
        return d[key]
    if (u, v) in d:
        return d[(u, v)]
    if (str(u), str(v)) in d:
        return d[(str(u), str(v))]
    return default


def dijkstra_shortest_path(G: dict, Dist: dict, start: str, goal: str) -> Tuple[List[str], float]:
    """Standard Dijkstra for shortest distance."""
    pq = [(0.0, start, None)]
    dist = {start: 0.0}
    prev = {start: None}
    while pq:
        d, u, p = heapq.heappop(pq)
        if d != dist.get(u, float("inf")):
            continue
        prev[u] = p
        if u == goal:
            break
        for v in G.get(u, []):
            w = edge_value(Dist, u, v)
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v, u))

    if goal not in dist:
        return [], float("inf")
    # reconstruct
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path, dist[goal]


def dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Return True if label a dominates b (<= in both and < in at least one)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def prune_labels(labels: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Keep only non-dominated labels (energy, dist)."""
    labels.sort()  # sort by energy
    pruned: List[Tuple[float, float]] = []
    best_dist = float("inf")
    for e, d in labels:
        if d < best_dist:
            pruned.append((e, d))
            best_dist = d
    return pruned


 # Constrained search with full parent reconstruction

def constrained_shortest_path_with_parents(
    G: dict,
    Dist: dict,
    Cost: dict,
    Coord: dict,
    start: str,
    goal: str,
    budget: float,
    use_astar: bool = False,
) -> Tuple[List[str], float, float]:
    def h(n: str) -> float:
        if not use_astar:
            return 0.0
        try:
            x1, y1 = Coord[n]
            x2, y2 = Coord[goal]
            return math.hypot(x1 - x2, y1 - y2)
        except Exception:
            return 0.0

    label_sets: Dict[str, List[Tuple[float, float]]] = {start: [(0.0, 0.0)]}
    pq: List[Label] = []
    heapq.heappush(pq, Label(h(start), 0.0, 0.0, start, None))

    parent_map: Dict[Tuple[str, float, float], Optional[Tuple[str, float, float]]] = {}
    parent_map[(start, 0.0, 0.0)] = None

    best_goal: Optional[Label] = None

    while pq:
        cur = heapq.heappop(pq)
        if cur.energy > budget:
            continue
        current_labels = label_sets.get(cur.node, [])
        dominated = False
        for e, d in current_labels:
            if dominates((e, d), (cur.energy, cur.dist)) and (e, d) != (cur.energy, cur.dist):
                dominated = True
                break
        if dominated:
            continue

        if cur.node == goal:
            if best_goal is None or cur.dist < best_goal.dist:
                best_goal = cur
                if use_astar:
                    if pq and pq[0].f >= best_goal.dist:
                        break
                else:
                    break

        for v in G.get(cur.node, []):
            w = edge_value(Dist, cur.node, v)
            c = edge_value(Cost, cur.node, v)
            ne = cur.energy + c
            if ne > budget:
                continue
            nd = cur.dist + w

            labels = label_sets.get(v, [])
            skip = False
            for e, d in labels:
                if dominates((e, d), (ne, nd)):
                    skip = True
                    break
            if skip:
                continue

            labels.append((ne, nd))
            label_sets[v] = prune_labels(labels)
            heapq.heappush(pq, Label(nd + h(v), nd, ne, v, (cur.node, cur.energy, cur.dist)))
            parent_map[(v, ne, nd)] = (cur.node, cur.energy, cur.dist)

    if best_goal is None:
        return [], float("inf"), float("inf")

    path = []
    key = (best_goal.node, best_goal.energy, best_goal.dist)
    while key is not None:
        node, e, d = key
        path.append(node)
        key = parent_map.get(key)
    path.reverse()
    return path, best_goal.dist, best_goal.energy


# ------------------------------
# Part 2: Grid World MDP/RL
# ------------------------------


ACTIONS = ["U", "D", "L", "R"]
MOVE = {
    "U": (0, 1),
    "D": (0, -1),
    "L": (-1, 0),
    "R": (1, 0),
}


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x <= 4 and 0 <= y <= 4


def step(state: Tuple[int, int], action: str, obstacles: set) -> Tuple[int, int]:
    dx, dy = MOVE[action]
    nx, ny = state[0] + dx, state[1] + dy
    if not in_bounds(nx, ny) or (nx, ny) in obstacles:
        return state
    return (nx, ny)


def transitions(state: Tuple[int, int], action: str, obstacles: set, stochastic: bool) -> List[Tuple[float, Tuple[int, int]]]:
    if not stochastic:
        return [(1.0, step(state, action, obstacles))]
    # stochastic model: 0.8 intended, 0.1 left/right
    if action == "U":
        left, right = "L", "R"
    elif action == "D":
        left, right = "R", "L"
    elif action == "L":
        left, right = "D", "U"
    else:
        left, right = "U", "D"
    return [
        (0.8, step(state, action, obstacles)),
        (0.1, step(state, left, obstacles)),
        (0.1, step(state, right, obstacles)),
    ]


def value_iteration(gamma: float, stochastic: bool) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
    obstacles = {(2, 1), (2, 3)}
    goal = (4, 4)
    states = [(x, y) for x in range(5) for y in range(5) if (x, y) not in obstacles]
    V = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            if s == goal:
                continue
            best = -1e9
            for a in ACTIONS:
                val = 0.0
                for p, ns in transitions(s, a, obstacles, stochastic):
                    reward = 10.0 if ns == goal else -1.0
                    val += p * (reward + gamma * V[ns])
                if val > best:
                    best = val
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < 1e-6:
            break

    policy = {}
    for s in states:
        if s == goal:
            policy[s] = "G"
            continue
        best_a = ACTIONS[0]
        best_v = -1e9
        for a in ACTIONS:
            val = 0.0
            for p, ns in transitions(s, a, obstacles, stochastic):
                reward = 10.0 if ns == goal else -1.0
                val += p * (reward + gamma * V[ns])
            if val > best_v:
                best_v = val
                best_a = a
        policy[s] = best_a
    return V, policy


def policy_iteration(gamma: float, stochastic: bool) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], str]]:
    obstacles = {(2, 1), (2, 3)}
    goal = (4, 4)
    states = [(x, y) for x in range(5) for y in range(5) if (x, y) not in obstacles]
    policy = {s: random.choice(ACTIONS) for s in states}
    policy[goal] = "G"

    while True:
        # policy evaluation
        V = {s: 0.0 for s in states}
        while True:
            delta = 0.0
            for s in states:
                if s == goal:
                    continue
                a = policy[s]
                val = 0.0
                for p, ns in transitions(s, a, obstacles, stochastic):
                    reward = 10.0 if ns == goal else -1.0
                    val += p * (reward + gamma * V[ns])
                delta = max(delta, abs(val - V[s]))
                V[s] = val
            if delta < 1e-6:
                break

        # policy improvement
        stable = True
        for s in states:
            if s == goal:
                continue
            old = policy[s]
            best_a = old
            best_v = -1e9
            for a in ACTIONS:
                val = 0.0
                for p, ns in transitions(s, a, obstacles, stochastic):
                    reward = 10.0 if ns == goal else -1.0
                    val += p * (reward + gamma * V[ns])
                if val > best_v:
                    best_v = val
                    best_a = a
            policy[s] = best_a
            if best_a != old:
                stable = False
        if stable:
            return V, policy


def mc_control(episodes: int, epsilon: float, stochastic: bool) -> Dict[Tuple[int, int], str]:
    obstacles = {(2, 1), (2, 3)}
    start = (0, 0)
    goal = (4, 4)
    states = [(x, y) for x in range(5) for y in range(5) if (x, y) not in obstacles]
    Q = {(s, a): 0.0 for s in states for a in ACTIONS}
    returns = {(s, a): [] for s in states for a in ACTIONS}

    def epsilon_greedy(s: Tuple[int, int]) -> str:
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        vals = [Q[(s, a)] for a in ACTIONS]
        return ACTIONS[vals.index(max(vals))]

    for _ in range(episodes):
        episode = []
        s = start
        while s != goal:
            a = epsilon_greedy(s)
            # sample transition
            r = random.random()
            probs = transitions(s, a, obstacles, stochastic)
            cum = 0.0
            for p, ns in probs:
                cum += p
                if r <= cum:
                    next_state = ns
                    break
            reward = 10.0 if next_state == goal else -1.0
            episode.append((s, a, reward))
            s = next_state

        G = 0.0
        visited = set()
        for s, a, reward in reversed(episode):
            G = reward + 0.9 * G
            if (s, a) not in visited:
                returns[(s, a)].append(G)
                Q[(s, a)] = sum(returns[(s, a)]) / len(returns[(s, a)])
                visited.add((s, a))

    policy = {}
    for s in states:
        if s == goal:
            policy[s] = "G"
            continue
        vals = [Q[(s, a)] for a in ACTIONS]
        policy[s] = ACTIONS[vals.index(max(vals))]
    return policy


def q_learning(episodes: int, epsilon: float, alpha: float, stochastic: bool) -> Dict[Tuple[int, int], str]:
    obstacles = {(2, 1), (2, 3)}
    start = (0, 0)
    goal = (4, 4)
    states = [(x, y) for x in range(5) for y in range(5) if (x, y) not in obstacles]
    Q = {(s, a): 0.0 for s in states for a in ACTIONS}

    def epsilon_greedy(s: Tuple[int, int]) -> str:
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        vals = [Q[(s, a)] for a in ACTIONS]
        return ACTIONS[vals.index(max(vals))]

    for _ in range(episodes):
        s = start
        while s != goal:
            a = epsilon_greedy(s)
            r = random.random()
            probs = transitions(s, a, obstacles, stochastic)
            cum = 0.0
            for p, ns in probs:
                cum += p
                if r <= cum:
                    next_state = ns
                    break
            reward = 10.0 if next_state == goal else -1.0
            best_next = max(Q[(next_state, a2)] for a2 in ACTIONS)
            Q[(s, a)] = Q[(s, a)] + alpha * (reward + 0.9 * best_next - Q[(s, a)])
            s = next_state

    policy = {}
    for s in states:
        if s == goal:
            policy[s] = "G"
            continue
        vals = [Q[(s, a)] for a in ACTIONS]
        policy[s] = ACTIONS[vals.index(max(vals))]
    return policy


def print_policy(policy: Dict[Tuple[int, int], str]) -> None:
    obstacles = {(2, 1), (2, 3)}
    for y in reversed(range(5)):
        row = []
        for x in range(5):
            if (x, y) in obstacles:
                row.append("#")
            else:
                row.append(policy.get((x, y), "?"))
        print(" ".join(row))


# ------------------------------
# Main
# ------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="SC3000/CZ3005 Lab Assignment 1")
    parser.add_argument("--part1", action="store_true", help="Run Part 1 (NYC instance)")
    parser.add_argument("--part2", action="store_true", help="Run Part 2 (Grid World)")
    parser.add_argument("--g", help="Path to graph dictionary G")
    parser.add_argument("--coord", help="Path to coordinate dictionary Coord")
    parser.add_argument("--dist", help="Path to distance dictionary Dist")
    parser.add_argument("--cost", help="Path to energy cost dictionary Cost")
    parser.add_argument("--start", default="1", help="Start node (default: 1)")
    parser.add_argument("--goal", default="50", help="Goal node (default: 50)")
    parser.add_argument("--budget", type=float, default=287932, help="Energy budget for tasks 2/3")
    parser.add_argument("--episodes", type=int, default=5000, help="Episodes for MC/Q-learning")
    args = parser.parse_args()

    if not args.part1 and not args.part2:
        args.part1 = True
        args.part2 = True

    if args.part1:
        print("[Part 1] Constrained shortest path")
        if args.g and args.coord and args.dist and args.cost:
            instance = (args.g, args.coord, args.dist, args.cost)
        else:
            instance = find_instance_files(".")
        if not instance:
            print("Instance files not found. Place G, Coord, Dist, Cost dictionaries in current folder.")
        else:
            g_path, coord_path, dist_path, cost_path = instance
            print(f"Loading instance files:\n  G={g_path}\n  Coord={coord_path}\n  Dist={dist_path}\n  Cost={cost_path}")
            G = load_dict(g_path)
            Coord = load_dict(coord_path)
            Dist = load_dict(dist_path)
            Cost = load_dict(cost_path)

            # Task 1
            path, dist = dijkstra_shortest_path(G, Dist, args.start, args.goal)
            print("Task 1 (Shortest path without energy constraint)")
            if path:
                print("Shortest path:", "->".join(path))
                print("Shortest distance:", dist)
            else:
                print("No path found.")

            # Task 2
            print("Task 2 (Uninformed search with energy budget - UCS)")
            path, dist, energy = constrained_shortest_path_with_parents(
                G, Dist, Cost, Coord, args.start, args.goal, args.budget, use_astar=False
            )
            if path:
                print("Shortest path:", "->".join(path))
                print("Shortest distance:", dist)
                print("Total energy cost:", energy)
            else:
                print("No feasible path found within budget.")

            # Task 3
            print("Task 3 (A* search with energy budget)")
            path, dist, energy = constrained_shortest_path_with_parents(
                G, Dist, Cost, Coord, args.start, args.goal, args.budget, use_astar=True
            )
            if path:
                print("Shortest path:", "->".join(path))
                print("Shortest distance:", dist)
                print("Total energy cost:", energy)
            else:
                print("No feasible path found within budget.")

    if args.part2:
        print("\n[Part 2] Grid World MDP/RL")
        print("Task 1: Value Iteration (deterministic)")
        V_vi, P_vi = value_iteration(0.9, stochastic=False)
        print_policy(P_vi)

        print("Task 1: Policy Iteration (deterministic)")
        V_pi, P_pi = policy_iteration(0.9, stochastic=False)
        print_policy(P_pi)

        print("Task 2: Monte Carlo Control (stochastic)")
        P_mc = mc_control(args.episodes, 0.1, stochastic=True)
        print_policy(P_mc)

        print("Task 3: Q-learning (stochastic)")
        P_ql = q_learning(args.episodes, 0.1, 0.1, stochastic=True)
        print_policy(P_ql)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
