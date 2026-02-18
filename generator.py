import numpy as np
import matplotlib.pyplot as plt

SEED = 1

"""
Label n random samples from d-dimensions, in the hypercube that has edge lengths 2u, 
and is centered about the origin
1. Randomly generate a vector which defines a d-1  dimension hyperplane (passes
through origin for simplicity)
2. Randomly generate a point (uniform distribution) in the hyper cube and classify it 
via the hyperplane
3. Output (x, y) where x are the n randomly generated points, and y[i] is the classification
of x[i] 
    y[i] is either 1 or 0

"""
def make_classification(n, d = 2, u = 1, rand_seed=SEED, gen_plot=False):
    # Random number generator
    rng = np.random.default_rng(seed=rand_seed) 

    # The coefficients defining the hyperplane
    a = rng.random(size=d) 

    # Vector a can have values [-1, 1)
    a = (a * 2) - 1 

    # Output variables
    x = rng.random(size=(n, d)) # n randomly generated points (uniform distribution [0, 1)) 
    y = np.zeros(n) # Labels 
    if gen_plot:
        zeros_xs = []
        zeros_ys = []
        ones_xs = []
        ones_ys = []

    # Scale x to the hypercube
    x = ((2 * u) * x)  - u 

    # print(f"d = {d}, n = {n}, u = {u}")
    # print(f"a = {a}")
    # print(f"x =\n{x}")

    # Classify n samples
    for i in range(n):
        total = 0
        for j in range(d):
            total += x[i, j] * a[j]
        if total > 0:
            #  Above hyper plane
            y[i] = 1
            if gen_plot:
                zeros_xs.append(x[i, 0])
                zeros_ys.append(x[i, 1])
        elif total == 0:
            # On hyper plane, regenerate:
            x[i] = rng.random(size=d)
            i -= 1
            print("regenerating point")
        elif gen_plot:
            ones_xs.append(x[i, 0])
            ones_ys.append(x[i, 1])
        # print(f"x[i] = {x[i]}, total = {total}, y[i] = {y[i]}")

    if gen_plot:
        #Generating Plot 
        if d > 2:
            print(f"WARNING: only plotting first two dimensions, since d ({d}) > 2")
        x_plot = np.linspace(-u, u, 100)
        if (a[1] == 0):
            print("Not drawing vertical line")
        else:
            y_plot = -(a[0] / a[1]) * x_plot # a1 * x + a2 * y = 0  -> y = -a1 * x / a2
            plt.plot(x_plot, y_plot, label='hyperplane', color='black')

        plt.scatter(zeros_xs, zeros_ys, c='r', s=10, marker='x') # 'c' for colors, 's' for size
        plt.scatter(ones_xs,   ones_ys, c='b', s=10, marker='o') # 'c' for colors, 's' for size

        # Set bounds of graph
        plt.ylim(-u, u)

        plt.show()    

    return (x, y)

def make_classification_test():
    make_classification(  10, rand_seed= 10, gen_plot=True)
    make_classification( 100, rand_seed=100, gen_plot=True)
    make_classification(1000, rand_seed=  3, gen_plot=True)
    make_classification(  50, rand_seed=  4, gen_plot=True)
    make_classification( 500, rand_seed=  5, gen_plot=True)
    # make_classification( 500, d = 3, rand_seed=  5, gen_plot=True)
    # make_classification( 500, d = 4, rand_seed=  5, gen_plot=True)
    # make_classification( 500, d = 5, rand_seed=  5, gen_plot=True)

if __name__ == "__main__":
    make_classification_test()
