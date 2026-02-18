import numpy as np
SEED = 1

# def classify_point()

"""
Label n random samples from d-dimensions, in the hypercube that has edge lengths 2u, 
and is centered about the origin
1. Randomly generate a hyperplane of dimension d-1 (passes through origin for simplicity)
2. Randomly generate a point (uniform distribution) in the hyper cube and classify it 
via the hyperplane
3. Output all randomly generated samples and their classifications
"""
def make_classification(n, d = 2, u = 1):
    # Random number generator
    rng = np.random.default_rng(seed=SEED) 

    # The coefficients defining the hyperplane
    a = rng.random(size=(d - 1)) 
    a = (a * 2) - 1

    # Output variables
    x = rng.random(size=(n, d)) # n randomly generated points (uniform distribution [0, 1)) 
    y = np.zeros(n)             # Labels 

    # Scale x to the hypercube
    x = ((2 * u) * x)  - u 
    # print(f"d = {d}, n = {n}, u = {u}")
    # print(f"a = {a}")
    # print(f"x =\n{x}")

    # # # x without it's last column (w/o last feature)
    # # trunc_x = x[:, :-1]
    # print(f"trunc_x =\n{trunc_x}")

    # Classify n samples
    for i in range(n):
        total = 0
        for j in range(d-1):
            total += x[i, j] * a[j]
        if total > x[i, d - 1]:
            #  Above hyper plane
            y[i] = 1
        # print(f"x[i] = {x[i]}, total = {total}, y[i] = {y[i]}")
  
    return (x, y)

if __name__ == "__main__":
    make_classification(10)
    make_classification(10, 4, 3)
    # make_classification(10, 4, 1)