import numpy as np
from plots.part_one import get_line_pts
import matplotlib.pyplot as plt

SEED = 1

"""
Label n random samples from d-dimensions, in the hypercube that has edge lengths 2u, 
and is centered about the origin
1. Randomly generate a vector which defines a d-1  dimension hyperplane (passes
through origin for simplicity)
2. Randomly generate a point (uniform distribution) in the hyper cube and classify it 
via the hyperplane
3. Output (x, y, a) where x are the n randomly generated points, and y[i] is the classification
of x[i] 
    y[i] is either 1 or 0
and a is the coefficient vector of the plane that defines the classification

"""
def make_classification(n, d = 2, u = 1, rand_seed=SEED, test_proportion=0.3):
    # Number of training data
    n_train = np.floor(n - (n * test_proportion)).astype(int)

    # Random number generator
    rng = np.random.default_rng(seed=rand_seed) 

    # The coefficients defining the hyperplane
    a = rng.random(size=d) 

    # Vector a can have values [-1, 1)
    a = (a * 2) - 1 

    # Output variables
    x = rng.random(size=(n, d)) # n randomly generated points (uniform distribution [0, 1)) 
    y = np.ones(n) # Labels 

    # Scale x to the hypercube
    x = ((2 * u) * x)  - u 

    # Classify n samples
    for i in range(n):
        total = 0
        for j in range(d):
            total += x[i, j] * a[j]
        if total < 0:
            #  Above hyper plane
            y[i] = -1
        
        elif total == 0:
            # On hyper plane, regenerate:
            x[i] = rng.random(size=d)
            i -= 1
            print("regenerating point")

    # Splitting x into test and training data
    return (x[:n_train], y[:n_train], x[n_train:], y[n_train:], a)

def make_plot(xs, ys, cs):
    fig, axs = plt.subplots(
        1,
        len(xs),
        figsize=(10, 3),
        sharey=True
    )
    for i in range(len(xs)):
        ax = axs[i]


        pts = xs[i]
        labels = ys[i]
        a = cs[i]
        u = 1

        # Transpose and separate x and y coordinate arrays 
        x, y = pts.T 

        (x_plot, y_plot) = get_line_pts(a=a, u=u)

        ax.plot(x_plot, y_plot, label='hyperplane', color='black')

        colors = np.where(labels == 1, 'red', 'blue')
        ax.scatter(x, y, c=colors, s=10, marker='x') # 'c' for colors, 's' for size

        # Set bounds of graph 
        ax.set_ylim(-u, u)

        # Square, hopefully 
        ax.set_aspect('equal', adjustable='box')
    
    # plt.title("Generated Points")
    plt.tight_layout()
    plt.savefig("plots/generated_points.svg")

    plt.show()

def make_classification_test():
    xs = []
    ys = []
    cs = []
    (x, y, _, _, a) = make_classification(100, rand_seed=10)
    xs.append(x)
    ys.append(y)
    cs.append(a)

    (x, y, _, _, a) = make_classification(100, rand_seed=100)
    xs.append(x)
    ys.append(y)
    cs.append(a)

    (x, y, _, _, a) = make_classification(100, rand_seed=  3)
    xs.append(x)
    ys.append(y)
    cs.append(a)

    (x, y, _, _, a) = make_classification(100, rand_seed=  4)
    xs.append(x)
    ys.append(y)
    cs.append(a)

    # (x, y, _, _, a) = make_classification( 500, rand_seed=  5)
    # xs.append(x)
    # ys.append(y)
    # cs.append(a)

    make_plot(xs, ys, cs)

if __name__ == "__main__":
    make_classification_test()
