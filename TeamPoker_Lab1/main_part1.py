import json
import heapq
import math


with open("G.json") as f:
    G = json.load(f)
with open("Dist.json") as f:
    Dist = json.load(f)
with open("Cost.json") as f:
    Cost = json.load(f)
with open("Coord.json") as f:
    Coord = json.load(f)

start_node = "1"
end_node = "50"
BUDGET = 287932


# Task 1 normal dijkstra, no energy constraint
def task1(start, end):

    dist = {}
    for node in G:
        dist[node] = float("inf")
    dist[start] = 0

    parent = {}
    parent[start] = None
    visited = set()

    pq = [(0, start)]

    while pq:
        cost, node = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        if node == end:
            break

        for nb in G[node]:
            edge = node + "," + nb
            if edge not in Dist:
                continue
            new_cost = cost + Dist[edge]
            if new_cost < dist[nb]:
                dist[nb] = new_cost
                parent[nb] = node
                heapq.heappush(pq, (new_cost, nb))

    # trace back the path
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()

    # sum energy cost along the path 
    total_energy = 0
    for i in range(len(path) - 1):
        key = path[i] + "," + path[i+1]
        total_energy += Cost.get(key, 0)

    return path, dist[end], total_energy


# Task 2: UCS but with energy budget constraint
# track (energy, dist) pairs per node and skip if already dominated

def task2(start, end, budget):

    # visited[node] stores list of (energy, dist) pairs we alr process
    visited = {}

    # parent maps (node, energy) to (parent_node, parent_energy) 
    parent = {}
    parent[(start, 0)] = None

    pq = [(0, 0, start)]  # (dist, energy, node)
    best_dist = float("inf")
    best_end_state = None

    while pq:
        d, e, node = heapq.heappop(pq)

        if node == end:
            if d < best_dist:
                best_dist = d
                best_end_state = (node, e)
            continue

        # check dominance 
        skip = False
        for (ve, vd) in visited.get(node, []):
            if ve <= e and vd <= d:
                skip = True
                break
        if skip:
            continue

        # update visited n remove dominated states
        old = visited.get(node, [])
        new_list = [(ve, vd) for (ve, vd) in old if not (e <= ve and d <= vd)]
        new_list.append((e, d))
        visited[node] = new_list

        for nb in G.get(node, []):
            edge = node + "," + nb
            if edge not in Dist or edge not in Cost:
                continue

            new_d = d + Dist[edge]
            new_e = e + Cost[edge]

            if new_e > budget:
                continue

            parent[(nb, new_e)] = (node, e)
            heapq.heappush(pq, (new_d, new_e, nb))

    if best_end_state is None:
        return None, float("inf"), float("inf")

    # reconstruct path from parent map
    path = []
    cur = best_end_state
    while cur is not None:
        path.append(cur[0])
        cur = parent.get(cur)
    path.reverse()

    total_energy = 0
    for i in range(len(path) - 1):
        total_energy += Cost[path[i] + "," + path[i+1]]

    return path, best_dist, total_energy


# straight line distance as heuristic 
def get_heuristic(node, goal):
    if node not in Coord or goal not in Coord:
        return 0
    x1, y1 = Coord[node]
    x2, y2 = Coord[goal]
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist


# Task 3:A* w energy budget
# f = g + h(n) as priority 
def task3(start, end, budget):

    visited = {}
    parent = {}
    parent[(start, 0)] = None

    h0 = get_heuristic(start, end)
    pq = [(0 + h0, 0, 0, start)]  # (f, g, energy, node)

    best_dist = float("inf")
    best_end_state = None

    while pq:
        f, d, e, node = heapq.heappop(pq)

        if node == end:
            if d < best_dist:
                best_dist = d
                best_end_state = (node, e)
            continue

        # dominance check
        skip = False
        for (ve, vd) in visited.get(node, []):
            if ve <= e and vd <= d:
                skip = True
                break
        if skip:
            continue

        old = visited.get(node, [])
        new_list = [(ve, vd) for (ve, vd) in old if not (e <= ve and d <= vd)]
        new_list.append((e, d))
        visited[node] = new_list

        for nb in G.get(node, []):
            edge = node + "," + nb
            if edge not in Dist or edge not in Cost:
                continue

            new_d = d + Dist[edge]
            new_e = e + Cost[edge]

            if new_e > budget:
                continue

            h = get_heuristic(nb, end)
            new_f = new_d + h

            parent[(nb, new_e)] = (node, e)
            heapq.heappush(pq, (new_f, new_d, new_e, nb))

    if best_end_state is None:
        return None, float("inf"), float("inf")

    path = []
    cur = best_end_state
    while cur is not None:
        path.append(cur[0])
        cur = parent.get(cur)
    path.reverse()

    total_energy = 0
    for i in range(len(path) - 1):
        total_energy += Cost[path[i] + "," + path[i+1]]

    return path, best_dist, total_energy


# main function to run all tasks and print results
def main():
    print("Task 1")
    path1, dist1, energy1 = task1(start_node, end_node)
    print("Shortest path:", "->".join(path1))
    print("Shortest distance:", dist1)
    print("Total energy cost:", energy1)

    print()
    print("Task 2")
    path2, dist2, energy2 = task2(start_node, end_node, BUDGET)
    print("Shortest path:", "->".join(path2))
    print("Shortest distance:", dist2)
    print("Total energy cost:", energy2)

    print()
    print("Task 3")
    path3, dist3, energy3 = task3(start_node, end_node, BUDGET)
    print("Shortest path:", "->".join(path3))
    print("Shortest distance:", dist3)
    print("Total energy cost:", energy3)
    
if __name__ == "__main__":
    main()
    
    