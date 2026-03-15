import json
import heapq
import math

def is_dominated(new_energy, new_dist, current_node_visited):
    # checks for better path
    for old_energy, old_dist in current_node_visited:
        if old_energy <= new_energy and old_dist <= new_dist:
            return True
    return False

def update_visited_states(new_energy, new_dist, current_node_visited):
    # keep okay old state
    updated_list = []
    for old_energy, old_dist in current_node_visited:
        if not (new_energy <= old_energy and new_dist <= old_dist):
            updated_list.append((old_energy, old_dist))
    
    updated_list.append((new_energy, new_dist))
    return updated_list

def reconstruct_path_and_energy(end_state, parent_map, cost_dict):
    path = []
    cur = end_state
    
    while cur is not None:
        path.append(cur[0])
        cur = parent_map.get(cur)
    path.reverse()

    total_energy = 0
    for i in range(len(path) - 1):
        edge_key = path[i] + "," + path[i+1]
        total_energy += cost_dict.get(edge_key, 0)

    return path, total_energy

def get_heuristic(node, goal, Coord):
    # straight line 
    if node not in Coord or goal not in Coord:
        return 0
    x1, y1 = Coord[node]
    x2, y2 = Coord[goal]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)



def task1(start, end, G, Dist, Cost):
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

        for nb in G.get(node, []):
            edge = node + "," + nb
            if edge not in Dist:
                continue
                
            new_cost = cost + Dist[edge]
            if new_cost < dist[nb]:
                dist[nb] = new_cost
                parent[nb] = node
                heapq.heappush(pq, (new_cost, nb))

    path, total_energy = reconstruct_path_and_energy((end, None), parent, Cost)
    return path, dist[end], total_energy


def task2(start, end, budget, G, Dist, Cost):
    visited = {}
    parent = {}
    parent[(start, 0)] = None

    pq = [(0, 0, start)]  # (dist, energy, node)
    best_dist = float("inf")
    best_end_state = None

    while pq:
        d, e, node = heapq.heappop(pq)

        # update dist
        if node == end:
            if d < best_dist:
                best_dist = d
                best_end_state = (node, e)
            continue

        if node in visited:
            if is_dominated(e, d, visited[node]):
                continue
            visited[node] = update_visited_states(e, d, visited[node])
        else:
            visited[node] = [(e, d)]

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

    path, total_energy = reconstruct_path_and_energy(best_end_state, parent, Cost)
    return path, best_dist, total_energy


def task3(start, end, budget, G, Dist, Cost, Coord):
    visited = {}
    parent = {}
    parent[(start, 0)] = None

    h0 = get_heuristic(start, end, Coord)
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

        if node in visited:
            if is_dominated(e, d, visited[node]):
                continue
            visited[node] = update_visited_states(e, d, visited[node])
        else:
            visited[node] = [(e, d)]

        for nb in G.get(node, []):
            edge = node + "," + nb
            if edge not in Dist or edge not in Cost:
                continue

            new_d = d + Dist[edge]
            new_e = e + Cost[edge]

            if new_e > budget:
                continue

            h = get_heuristic(nb, end, Coord)
            new_f = new_d + h

            parent[(nb, new_e)] = (node, e)
            heapq.heappush(pq, (new_f, new_d, new_e, nb))

    if best_end_state is None:
        return None, float("inf"), float("inf")

    path, total_energy = reconstruct_path_and_energy(best_end_state, parent, Cost)
    return path, best_dist, total_energy


def load_data():
    with open("G.json") as f:
        G = json.load(f)
    with open("Dist.json") as f:
        Dist = json.load(f)
    with open("Cost.json") as f:
        Cost = json.load(f)
    with open("Coord.json") as f:
        Coord = json.load(f)
    return G, Dist, Cost, Coord


def main():
    
    G, Dist, Cost, Coord = load_data()
    start_node = "1"
    end_node = "50"
    budget = 287932

    print("Task 1")
    path1, dist1, energy1 = task1(start_node, end_node, G, Dist, Cost)
    print("Shortest path:", "->".join(path1) if path1 else "None")
    print("Shortest distance:", dist1)
    print("Total energy cost:", energy1)


    print("\nTask 2")
    path2, dist2, energy2 = task2(start_node, end_node, budget, G, Dist, Cost)
    print("Shortest path:", "->".join(path2) if path2 else "None")
    print("Shortest distance:", dist2)
    print("Total energy cost:", energy2)

    print("\nTask 3")
    path3, dist3, energy3 = task3(start_node, end_node, budget, G, Dist, Cost, Coord)
    print("Shortest path:", "->".join(path3) if path3 else "None")
    print("Shortest distance:", dist3)
    print("Total energy cost:", energy3)
    
if __name__ == "__main__":
    main()