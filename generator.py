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
3. Output (x, y, a) where x are the n randomly generated points, and y[i] is the classification
of x[i] 
    y[i] is either 1 or 0
and a is the coefficient vector of the plane that defines the classification

"""
def make_classification(n, d = 2, u = 1, rand_seed=SEED):
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

    return (x, y, a)

"""
Generate points on a line
"""
def get_line_pts(a, u=1):
    xs = np.linspace(-u, u, 100)
    if (a[1] == 0):
        print("ERROR: VERTICAL LINE")
        return 0
    ys = -(a[0] / a[1]) * xs # a1 * x + a2 * y = 0  -> y = -a1 * x / a2
    return (xs, ys)    

"""
generate a plot given points, labels, and the coefficients defining a hyper plane 
NOTE: Will only work for 2D
"""
def gen_plot(pts, labels, a, u=1):
    # Transpose separate x and y coordinate arrays
    x, y = pts.T 

    (x_plot, y_plot) = get_line_pts(a, u)

    plt.plot(x_plot, y_plot, label='hyperplane', color='black')

    colors = np.where(labels == 1, 'red', 'blue')
    plt.scatter(x, y, c=colors, s=10, marker='x') # 'c' for colors, 's' for size

    # Set bounds of graph
    plt.ylim(-u, u)

    plt.show()

def make_classification_test():
    (x, y, a) = make_classification(  10, rand_seed= 10)
    gen_plot(x, y, a)

    (x, y, a) = make_classification( 100, rand_seed=100)
    gen_plot(x, y, a)

    (x, y, a) = make_classification(1000, rand_seed=  3)
    gen_plot(x, y, a)

    (x, y, a) = make_classification(  50, rand_seed=  4)
    gen_plot(x, y, a)

    (x, y, a) = make_classification( 500, rand_seed=  5)
    gen_plot(x, y, a)


if __name__ == "__main__":
    make_classification_test()
