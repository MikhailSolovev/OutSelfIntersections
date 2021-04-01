import random as r
import matplotlib.pyplot as plt
import numpy as np
import time

# Constants
S = 5
N = np.arange(20, 85, 5)
P = 3


def basis(field):  # generate one chain without intersections length of S
    new_field = np.array(field)
    x, y, z = field[-1]
    dxp, dyp, dzp = 0, 0, 0
    for _ in range(S):
        dx, dy, dz = r.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
        while (dx, dy, dz) == (-dxp, -dyp, -dzp):
            dx, dy, dz = r.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
        x, y, z = x + dx, y + dy, z + dz
        if any(np.equal(new_field, [x, y, z]).all(1)):
            return None
        else:
            new_field = np.concatenate((new_field, np.array([(x, y, z)])), axis=0)
        dxp, dyp, dzp = dx, dy, dz
    return new_field


def step(field):  # generate other chains, total amount P
    chains = []
    for _ in range(P):
        sample = basis(field)
        if sample is not None:
            chains.append(sample)
    return np.array(chains)


def main():
    r2 = []
    chains = step([(0, 0, 0)])
    for i in range(N[-1] // S - 1):
        start = time.time()
        new_chains = []
        for chain in chains:
            for c in step(chain):
                new_chains.append(c)
            chains = np.array(new_chains)
        print("[DONE]", i, "[TIME]", round(time.time() - start, 4), 'sec')
        # average squared radius
        if i >= 2:
            sum_ = 0
            for chain in chains:
                sum_ += chain[-1, 0] ** 2 + chain[-1, 1] ** 2 + chain[-1, 2] ** 2
            r2.append(sum_ / len(chains))
    return np.array(r2)


r2 = main()

# Graphs

plt.plot(N, r2, 'o')
plt.xlabel('N')
plt.ylabel('<r^2>')
plt.title(f'S: {S} N: [{N[0]}, {N[-1]}] P: {P}')
plt.show()

# Double logarithm
N_log = np.log(N)
r2_log = np.log(r2)
# MNK
k, b = np.polyfit(N_log, r2_log, 1)
correlation_matrix = np.corrcoef(N_log, r2_log)
correlation_xy = correlation_matrix[0, 1]
r_squared = correlation_xy**2


print('k:', k, 'b:', b)
print('r^2:', r_squared)

line = lambda x: k*x + b

plt.plot(N_log, r2_log, 'o')
plt.plot(N_log, line(N_log), 'purple')
plt.xlabel('ln(N)')
plt.ylabel('ln(<r^2>)')
plt.title(f'S: {S} N: [{N[0]}, {N[-1]}] P: {P}')
plt.show()
