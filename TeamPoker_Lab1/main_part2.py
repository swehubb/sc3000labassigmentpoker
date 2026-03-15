import random
import math

n = 5
start = (0, 0)
goal = (4, 4)
obstacles = {(2, 1), (2, 3)}
actions = ["U", "D", "L", "R"]

gamma = 0.9
epsilon = 0.1
alpha = 0.1
episodes = 20000
max_steps = 500

p_intended = 0.8
p_side = 0.1  

def in_bounds(s):
    x, y = s
    return 0 <= x < n and 0 <= y < n

def is_blocked(s):
    return s in obstacles

def is_goal(s):
    return s == goal

def get_all_states():
    states = []
    for x in range(n):
        for y in range(n):
            if not is_blocked((x, y)):
                states.append((x, y))
    return states

def move(s, a):
    if is_goal(s):
        return s
    x, y = s
    if a == "U":
        nx, ny = x, y + 1
    elif a == "D":
        nx, ny = x, y - 1
    elif a == "L":
        nx, ny = x - 1, y
    else:  
        nx, ny = x + 1, y

    ns = (nx, ny)
    if not in_bounds(ns) or is_blocked(ns):
        return s
    return ns

def det_step(s, a):
    if is_goal(s):
        return s, 0, True
    ns = move(s, a)
    if ns == goal:
        return ns, 10, True
    return ns, -1, False

def stoch_step(s, a):
    if is_goal(s):
        return s, 0, True

    if a in ("U", "D"):
        sides = ("L", "R")
    else:
        sides = ("U", "D")

    r = random.random()
    if r < p_intended:
        actual = a
    elif r < p_intended + p_side:
        actual = sides[0]
    else:
        actual = sides[1]

    return det_step(s, actual)

def action_char(a):
    if a is None:
        return "."
    return a

def print_values(V, title):
    print(title)
    for y in reversed(range(n)):
        row = []
        for x in range(n):
            s = (x, y)
            if s in obstacles:
                row.append("   X   ")
            elif s == goal:
                row.append("   G   ")
            else:
                row.append(f"{V.get(s, 0.0):6.2f} ")
        print(" ".join(row))
    print()

def print_policy(pi, title):
    print(title)
    for y in reversed(range(n)):
        row = []
        for x in range(n):
            s = (x, y)
            if s in obstacles:
                row.append("X")
            elif s == goal:
                row.append("G")
            else:
                row.append(action_char(pi.get(s)))
        print("  ".join(row))
    print()

def compare_policies(pi1, pi2, label1, label2):
    diff = []
    for s in get_all_states():
        if is_goal(s):
            continue
        if pi1.get(s) != pi2.get(s):
            diff.append(s)

    if not diff:
        print(f"{label1} and {label2}: same policy")
    else:
        print(f"{label1} vs {label2}: differ at {len(diff)} states")
        for s in diff:
            print(f"  {s}:  {label1}={pi1.get(s)}  {label2}={pi2.get(s)}")
    print()

# value iteration
def value_iteration():
    V = {}
    for s in get_all_states():
        V[s] = 0.0

    for i in range(10000):
        delta = 0
        for s in get_all_states():
            if is_goal(s):
                continue

            old_v = V[s]
            best = float("-inf") 
            
            for a in actions:
                ns, r, done = det_step(s, a)
                if done:
                    q = r
                else:
                    q = r + gamma * V[ns]
                if q > best:
                    best = q

            V[s] = best
            
            
            if abs(V[s] - old_v) > delta:
                delta = abs(V[s] - old_v)

        if delta < 1e-6:
            break

    pi = {}
    for s in get_all_states():
        if is_goal(s):
            continue
        best_a = None
        best_q = float("-inf")
        for a in actions:
            ns, r, done = det_step(s, a)
            if done:
                q = r
            else:
                q = r + gamma * V[ns]
            if q > best_q:
                best_q = q
                best_a = a
        pi[s] = best_a

    return V, pi

# policy iteration
def policy_eval(pi):
    V = {}
    for s in get_all_states():
        V[s] = 0.0

    for _ in range(10000):
        delta = 0
        for s in get_all_states():
            if is_goal(s):
                continue
            old_v = V[s]
            a = pi[s]
            ns, r, done = det_step(s, a)
            if done:
                v = r
            else:
                v = r + gamma * V[ns]
            V[s] = v
            
            if abs(V[s] - old_v) > delta:
                delta = abs(V[s] - old_v)
                
        if delta < 1e-6:
            break

    return V

def policy_iteration():
    pi = {}
    for s in get_all_states():
        if not is_goal(s):
            pi[s] = "U"

    while True:
        V = policy_eval(pi)
        changed = False
        
        for s in get_all_states():
            if is_goal(s):
                continue

            old_a = pi[s]
            best_a = None
            best_q = float("-inf")

            for a in actions:
                ns, r, done = det_step(s, a)
                if done:
                    q = r
                else:
                    q = r + gamma * V[ns]
                if q > best_q:
                    best_q = q
                    best_a = a

            pi[s] = best_a
            if best_a != old_a:
                changed = True

        if not changed:
            break

    return V, pi

def eps_greedy(Q, s, eps):
    if random.random() < eps:
        return random.choice(actions)
        
    best_a = actions[0]
    best_q = Q[(s, actions[0])]
    for a in actions:
        q = Q[(s, a)]
        if q > best_q:
            best_q = q
            best_a = a
    return best_a

def extract_policy(Q):
    pi = {}
    for s in get_all_states():
        if is_goal(s):
            continue
        best_a = actions[0]
        best_q = Q[(s, actions[0])]
        for a in actions:
            q = Q[(s, a)]
            if q > best_q:
                best_q = q
                best_a = a
        pi[s] = best_a
    return pi

def monte_carlo(num_episodes):
    Q = {}
    returns = {}  
    
 
    for s in get_all_states():
        for a in actions:
            Q[(s, a)] = 0.0
            returns[(s, a)] = []

    for ep in range(num_episodes):
        s = start
        episode = []

        for _ in range(max_steps):
            if is_goal(s):
                break
            a = eps_greedy(Q, s, epsilon)
            ns, r, done = stoch_step(s, a)
            episode.append((s, a, r))
            s = ns
            if done:
                break

        T = len(episode)
        for t in range(T):
            st, at, _ = episode[t]

            first_visit = True
            for j in range(t):
                if episode[j][0] == st and episode[j][1] == at:
                    first_visit = False
                    break
            if not first_visit:
                continue

            # explicitly looping powers for gamma
            G = 0.0
            power = 0
            for k in range(t, T):
                G += (gamma ** power) * episode[k][2]
                power += 1

            returns[(st, at)].append(G)
            
            # calculate average
            total_sum = 0
            for value in returns[(st, at)]:
                total_sum += value
            Q[(st, at)] = total_sum / len(returns[(st, at)])

    return Q

def q_learning(num_episodes):
    Q = {}
    
    # initialise
    for s in get_all_states():
        for a in actions:
            Q[(s, a)] = 0.0

    for ep in range(num_episodes):
        s = start

        for _ in range(max_steps):
            if is_goal(s):
                break

            a = eps_greedy(Q, s, epsilon)
            ns, r, done = stoch_step(s, a)

            old = Q[(s, a)]
            
            if done:
                target = r
            else:
                best_next = float("-inf")
                for na in actions:
                    if Q[(ns, na)] > best_next:
                        best_next = Q[(ns, na)]
                target = r + gamma * best_next

            Q[(s, a)] = old + alpha * (target - old)
            s = ns

            if done:
                break

    return Q

def V_from_Q(Q):
    V = {}
    for s in get_all_states():
        if is_goal(s):
            V[s] = 0.0
        else:
            best_val = float("-inf")
            for a in actions:
                if Q[(s, a)] > best_val:
                    best_val = Q[(s, a)]
            V[s] = best_val
    return V

random.seed(0)

print("Task 1")
V_vi, pi_vi = value_iteration()
V_pit, pi_pit = policy_iteration()

print_values(V_vi, "Value Iteration - state values")
print_policy(pi_vi, "Value Iteration - policy")

print_values(V_pit, "Policy Iteration - state values")
print_policy(pi_pit, "Policy Iteration - policy")

compare_policies(pi_vi, pi_pit, "Value Iter", "Policy Iter")

print()
print("Task 2")
Q_mc = monte_carlo(episodes)
pi_mc = extract_policy(Q_mc)
V_mc = V_from_Q(Q_mc)

print_values(V_mc, "Monte Carlo - state values")
print_policy(pi_mc, "Monte Carlo - policy")

_, pi_opt = value_iteration()
compare_policies(pi_mc, pi_opt, "Monte Carlo", "Optimal (VI)")

print()
print("Task 3")
Q_ql = q_learning(episodes)
pi_ql = extract_policy(Q_ql)
V_ql = V_from_Q(Q_ql)

print_values(V_ql, "Q-learning - state values")
print_policy(pi_ql, "Q-learning - policy")

random.seed(0)
Q_mc2 = monte_carlo(episodes)
pi_mc2 = extract_policy(Q_mc2)

compare_policies(pi_ql, pi_mc2, "Q-learning", "Monte Carlo")
compare_policies(pi_ql, pi_opt, "Q-learning", "Optimal (VI)")