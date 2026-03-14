# main.py
# CZ3005 Lab Assignment 1 - Part 1 (NYC)
# Task 1: shortest path without energy constraint (Dijkstra on distance)
# Task 2: uninformed search under energy budget (UCS-style label-setting with dominance pruning)
# Task 3: A* under energy budget (label-setting with f=g+h, h from Coord straight-line distance)

import json
import math
import heapq
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

BUDGET = 287_932  # for Task 2 & 3
START = "1"
GOAL = "50"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_str_node(x: Any) -> str:
    # JSON keys are always strings, but values (neighbors) might be int or str.
    return str(x)


def normalize_graph(G_raw: Dict) -> Dict[str, List[str]]:
    G: Dict[str, List[str]] = {}
    for k, nbrs in G_raw.items():
        sk = to_str_node(k)
        G[sk] = [to_str_node(v) for v in nbrs]
    return G


def normalize_edge_dict(d: Dict) -> Dict[str, float]:
    # Dist/Cost keys like "v,w" -> number
    out: Dict[str, float] = {}
    for k, v in d.items():
        out[str(k)] = float(v)
    return out


def normalize_coord(C_raw: Dict) -> Dict[str, Tuple[float, float]]:
    C: Dict[str, Tuple[float, float]] = {}
    for k, v in C_raw.items():
        sk = to_str_node(k)
        # v could be [x,y] or (x,y)
        C[sk] = (float(v[0]), float(v[1]))
    return C


def fmt_number(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else str(x)


def dijkstra_shortest_path(
    G: Dict[str, List[str]],
    Dist: Dict[str, float],
    Cost: Dict[str, float],
    start: str,
    goal: str
) -> Tuple[Optional[List[str]], float, float]:
    """Task 1: classic Dijkstra on distance only. Also sums energy along the chosen path for printing."""
    pq: List[Tuple[float, str]] = [(0.0, start)]
    best: Dict[str, float] = {start: 0.0}
    parent: Dict[str, Optional[str]] = {start: None}

    while pq:
        d, u = heapq.heappop(pq)
        if d != best.get(u, math.inf):
            continue
        if u == goal:
            break
        for v in G.get(u, []):
            key = f"{u},{v}"
            w = Dist.get(key)
            if w is None:
                continue
            nd = d + w
            if nd < best.get(v, math.inf):
                best[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if goal not in best:
        return None, math.inf, math.inf

    # reconstruct path
    path: List[str] = []
    cur: Optional[str] = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()

    # compute energy for printing
    energy = 0.0
    for i in range(len(path) - 1):
        energy += Cost.get(f"{path[i]},{path[i+1]}", 0.0)

    return path, best[goal], energy


@dataclass(frozen=True)
class Label:
    node: str
    dist: float   # g cost (distance)
    energy: float # resource cost
    parent_id: Optional[int]


def is_dominated(new_e: float, new_d: float, existing: List[Tuple[float, float]]) -> bool:
    """
    Returns True if (new_e,new_d) is dominated by any existing (e,d),
    i.e., exists (e<=new_e and d<=new_d) with at least one strict.
    """
    for e, d in existing:
        if e <= new_e and d <= new_d and (e < new_e or d < new_d):
            return True
    return False


def prune_frontier(frontier: List[Tuple[float, float]], new_e: float, new_d: float) -> List[Tuple[float, float]]:
    """
    Remove labels dominated by (new_e,new_d): i.e., those with e>=new_e and d>=new_d.
    Keep non-dominated ones, then add the new one.
    """
    kept = []
    for e, d in frontier:
        if not (e >= new_e and d >= new_d and (e > new_e or d > new_d)):
            kept.append((e, d))
    kept.append((new_e, new_d))
    return kept


def reconstruct_path(labels: Dict[int, Label], best_goal_id: int) -> List[str]:
    path: List[str] = []
    cur_id: Optional[int] = best_goal_id
    while cur_id is not None:
        lab = labels[cur_id]
        path.append(lab.node)
        cur_id = lab.parent_id
    path.reverse()
    return path


def rcsp_ucs(
    G: Dict[str, List[str]],
    Dist: Dict[str, float],
    Cost: Dict[str, float],
    start: str,
    goal: str,
    budget: float
) -> Tuple[Optional[List[str]], float, float]:
    """
    Task 2: Uninformed search under energy constraint (resource-constrained shortest path).
    UCS-style label-setting: priority by dist (g).
    Each node keeps a Pareto frontier of (energy, dist); dominance pruning reduces blow-up.
    """
    pq: List[Tuple[float, int]] = []  # (priority=g, label_id)
    labels: Dict[int, Label] = {}
    frontier: Dict[str, List[Tuple[float, float]]] = {}  # node -> list of (energy,dist)

    next_id = 0
    start_label = Label(node=start, dist=0.0, energy=0.0, parent_id=None)
    labels[next_id] = start_label
    frontier[start] = [(0.0, 0.0)]
    heapq.heappush(pq, (0.0, next_id))
    next_id += 1

    best_goal_id: Optional[int] = None
    best_goal_dist = math.inf

    while pq:
        g, lid = heapq.heappop(pq)
        lab = labels[lid]

        # stale check: if this label is already worse than known best goal, can stop early-ish
        if lab.dist != g:
            continue
        if lab.node == goal:
            if lab.dist < best_goal_dist:
                best_goal_dist = lab.dist
                best_goal_id = lid
            # We can continue a bit, but UCS property + pruning often makes this sufficient.
            # We'll keep going to ensure no other feasible label has smaller distance.
            # Optional early break:
            # break

        for v in G.get(lab.node, []):
            d_key = f"{lab.node},{v}"
            c_key = f"{lab.node},{v}"
            w = Dist.get(d_key)
            c = Cost.get(c_key)
            if w is None or c is None:
                continue

            new_d = lab.dist + w
            new_e = lab.energy + c
            if new_e > budget:
                continue

            f_list = frontier.get(v, [])
            if is_dominated(new_e, new_d, f_list):
                continue

            # accept new label; prune dominated ones at v
            frontier[v] = prune_frontier(f_list, new_e, new_d)

            new_label = Label(node=v, dist=new_d, energy=new_e, parent_id=lid)
            labels[next_id] = new_label
            heapq.heappush(pq, (new_d, next_id))
            next_id += 1

    if best_goal_id is None:
        return None, math.inf, math.inf

    path = reconstruct_path(labels, best_goal_id)
    best_lab = labels[best_goal_id]
    return path, best_lab.dist, best_lab.energy


def heuristic(coord: Dict[str, Tuple[float, float]], a: str, b: str) -> float:
    """Straight-line distance lower bound (admissible for distance if coords are Euclidean-like)."""
    if a not in coord or b not in coord:
        return 0.0
    ax, ay = coord[a]
    bx, by = coord[b]
    return math.hypot(ax - bx, ay - by)


def rcsp_astar(
    G: Dict[str, List[str]],
    Dist: Dict[str, float],
    Cost: Dict[str, float],
    Coord: Dict[str, Tuple[float, float]],
    start: str,
    goal: str,
    budget: float
) -> Tuple[Optional[List[str]], float, float]:
    """
    Task 3: A* under energy constraint.
    Label-setting + dominance pruning, but priority is f = g + h.
    """
    pq: List[Tuple[float, float, int]] = []  # (f, g, label_id)
    labels: Dict[int, Label] = {}
    frontier: Dict[str, List[Tuple[float, float]]] = {}

    next_id = 0
    start_label = Label(node=start, dist=0.0, energy=0.0, parent_id=None)
    labels[next_id] = start_label
    frontier[start] = [(0.0, 0.0)]
    h0 = heuristic(Coord, start, goal)
    heapq.heappush(pq, (h0, 0.0, next_id))
    next_id += 1

    best_goal_id: Optional[int] = None
    best_goal_dist = math.inf

    while pq:
        f, g, lid = heapq.heappop(pq)
        lab = labels[lid]
        if lab.dist != g:
            continue

        # A* stopping condition: if we already found a goal with dist <= current best possible f
        if best_goal_id is not None and f >= best_goal_dist:
            break

        if lab.node == goal:
            if lab.dist < best_goal_dist:
                best_goal_dist = lab.dist
                best_goal_id = lid
            continue

        for v in G.get(lab.node, []):
            d_key = f"{lab.node},{v}"
            c_key = f"{lab.node},{v}"
            w = Dist.get(d_key)
            c = Cost.get(c_key)
            if w is None or c is None:
                continue

            new_g = lab.dist + w
            new_e = lab.energy + c
            if new_e > budget:
                continue

            f_list = frontier.get(v, [])
            if is_dominated(new_e, new_g, f_list):
                continue

            frontier[v] = prune_frontier(f_list, new_e, new_g)

            new_label = Label(node=v, dist=new_g, energy=new_e, parent_id=lid)
            labels[next_id] = new_label
            h = heuristic(Coord, v, goal)
            heapq.heappush(pq, (new_g + h, new_g, next_id))
            next_id += 1

    if best_goal_id is None:
        return None, math.inf, math.inf

    path = reconstruct_path(labels, best_goal_id)
    best_lab = labels[best_goal_id]
    return path, best_lab.dist, best_lab.energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=[1, 2, 3], default=1, help="Task number: 1/2/3")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    G = normalize_graph(load_json(base / "G.json"))
    Dist = normalize_edge_dict(load_json(base / "Dist.json"))
    Cost = normalize_edge_dict(load_json(base / "Cost.json"))
    Coord = normalize_coord(load_json(base / "Coord.json"))

    start, goal = START, GOAL

    if args.task == 1:
        path, dist, energy = dijkstra_shortest_path(G, Dist, Cost, start, goal)
    elif args.task == 2:
        path, dist, energy = rcsp_ucs(G, Dist, Cost, start, goal, BUDGET)
    else:
        path, dist, energy = rcsp_astar(G, Dist, Cost, Coord, start, goal, BUDGET)

    if path is None:
        print(f"Shortest path: None")
        print(f"Shortest distance: inf")
        print(f"Total energy cost: inf")
        return

    print("Shortest path:", "->".join(path))
    print("Shortest distance:", fmt_number(dist))
    print("Total energy cost:", fmt_number(energy))


if __name__ == "__main__":
    main()