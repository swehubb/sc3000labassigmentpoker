import random
import math

# ====== Environment spec ======
N = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = {(2, 1), (2, 3)}
ACTIONS = ["U", "D", "L", "R"]
GAMMA = 0.9

P_INTENDED = 0.8
P_LEFT = 0.1
P_RIGHT = 0.1

EPSILON = 0.1
ALPHA = 0.1
EPISODES = 20000
MAX_STEPS_PER_EPISODE = 500


# ====== Basic environment helpers ======
def in_bounds(s):
    x, y = s
    return 0 <= x < N and 0 <= y < N


def is_terminal(s):
    return s == GOAL


def is_blocked(s):
    return s in OBSTACLES


def all_states():
    states = []
    for x in range(N):
        for y in range(N):
            s = (x, y)
            if not is_blocked(s):
                states.append(s)
    return states


def move_once(s, a):
    if is_terminal(s):
        return s

    x, y = s
    if a == "U":
        ns = (x, y + 1)
    elif a == "D":
        ns = (x, y - 1)
    elif a == "L":
        ns = (x - 1, y)
    else:
        ns = (x + 1, y)

    if not in_bounds(ns) or is_blocked(ns):
        return s
    return ns


def step_deterministic(s, a):
    if is_terminal(s):
        return s, 0.0, True

    ns = move_once(s, a)
    if ns == GOAL:
        return ns, 10.0, True
    return ns, -1.0, False


def perpendicular_actions(a):
    if a in ("U", "D"):
        return ("L", "R")
    return ("U", "D")


def deterministic_transition_model(s, a):
    ns, r, done = step_deterministic(s, a)
    return [(1.0, ns, r, done)]


def sample_step(s, a):
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
def policy_arrow(a):
    if a is None:
        return "·"
    return {"U": "↑", "D": "↓", "L": "←", "R": "→"}[a]


def print_grid_values(V, title):
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


def print_grid_policy(pi, title):
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


def compare_policies(pi1, pi2, name1, name2):
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


# ====== Task 1: Value Iteration ======
def value_iteration(theta=1e-6, max_iters=10000):
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

    pi = {}
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


# ====== Task 1: Policy Iteration ======
def policy_evaluation(pi, theta=1e-6, max_iters=10000):
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


def policy_iteration():
    pi = {}
    for s in all_states():
        if not is_terminal(s):
            pi[s] = "U"

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
def epsilon_greedy_action(Q, s, epsilon):
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


def greedy_policy_from_Q(Q):
    pi = {}
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


def V_from_Q(Q):
    V = {}
    for s in all_states():
        if is_terminal(s):
            V[s] = 0.0
        else:
            V[s] = max(Q.get((s, a), 0.0) for a in ACTIONS)
    return V


# ====== Task 2: Monte Carlo Control ======
def monte_carlo_control(episodes):
    Q = {}
    returns_sum = {}
    returns_cnt = {}

    for _ in range(episodes):
        s = START
        episode = []

        for _ in range(MAX_STEPS_PER_EPISODE):
            if is_terminal(s):
                break

            a = epsilon_greedy_action(Q, s, EPSILON)
            ns, r, done = sample_step(s, a)
            episode.append((s, a, r))
            s = ns

            if done:
                break

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
def q_learning(episodes):
    Q = {}

    for _ in range(episodes):
        s = START

        for _ in range(MAX_STEPS_PER_EPISODE):
            if is_terminal(s):
                break

            a = epsilon_greedy_action(Q, s, EPSILON)
            ns, r, done = sample_step(s, a)

            old_q = Q.get((s, a), 0.0)
            best_next = 0.0 if done else max(Q.get((ns, na), 0.0) for na in ACTIONS)
            target = r + GAMMA * best_next
            Q[(s, a)] = old_q + ALPHA * (target - old_q)

            s = ns
            if done:
                break

    pi = greedy_policy_from_Q(Q)
    return Q, pi


def main():
    random.seed(0)

    print("Task 1")
    V_vi, pi_vi = value_iteration()
    V_pi, pi_pi = policy_iteration()

    print_grid_values(V_vi, "Value Iteration: V(s)")
    print_grid_policy(pi_vi, "Value Iteration: policy")
    print_grid_values(V_pi, "Policy Iteration: V(s)")
    print_grid_policy(pi_pi, "Policy Iteration: policy")
    compare_policies(pi_vi, pi_pi, "Value Iteration", "Policy Iteration")

    print("Task 2")
    Q_mc, pi_mc = monte_carlo_control(EPISODES)
    V_mc = V_from_Q(Q_mc)
    _, pi_opt = value_iteration()

    print_grid_values(V_mc, f"Monte Carlo Control (eps={EPSILON}): V(s)")
    print_grid_policy(pi_mc, "Monte Carlo Control: policy")
    compare_policies(pi_mc, pi_opt, "Monte Carlo", "Optimal (VI)")

    print("Task 3")
    Q_ql, pi_ql = q_learning(EPISODES)
    V_ql = V_from_Q(Q_ql)
    _, pi_opt = value_iteration()
    _, pi_mc = monte_carlo_control(EPISODES)

    print_grid_values(V_ql, f"Q-learning (eps={EPSILON}, alpha={ALPHA}): V(s)")
    print_grid_policy(pi_ql, "Q-learning: policy")
    compare_policies(pi_ql, pi_mc, "Q-learning", "Monte Carlo")
    compare_policies(pi_ql, pi_opt, "Q-learning", "Optimal (VI)")


if __name__ == "__main__":
    main()