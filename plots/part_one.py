import numpy as np
import matplotlib.pyplot as plt

"""
Generate points on a line of form:
a1 * x + a2 * y + b = c
"""
def get_line_pts(a, u=1, c=0, b=0):
    xs = np.linspace(-u, u, 100)
    if (a[1] == 0):
        print("ERROR: VERTICAL LINE")
        return 0
    ys = (c - b - a[0]* xs) / a[1]# a1 * x + a2 * y + b = c  -> y = (c - b - a1 * x) / a2
    return (xs, ys)    

"""
generate a plot given points, labels, and the coefficients defining a hyper plane 
NOTE: Will only work for 2D
"""
def gen_plot(pts, labels, a, u=1):
    # Transpose separate x and y coordinate arrays
    x, y = pts.T 

    (x_plot, y_plot) = get_line_pts(a=a, u=u)

    plt.plot(x_plot, y_plot, label='hyperplane', color='black')

    colors = np.where(labels == 1, 'red', 'blue')
    plt.scatter(x, y, c=colors, s=10, marker='x') # 'c' for colors, 's' for size

    # Set bounds of graph
    plt.ylim(-u, u)

    plt.show()


"""
Generate a plot displaying the points as labeled, as well as the margin generated via an SVC
given the terminal weights, and bias
pts    - sample points
labels - SVC predicted labels (y hat)
w      - weights of SVC upon termination
b      - the bias of SVC upon termination
"""
def display_margins(pts, labels, w, b, u=1):
    # Transpose separate x and y coordinate arrays
    x, y = pts.T 

    (x_plane, y_plane) = get_line_pts(w, u=u, b=b)
    (x_upper, y_upper) = get_line_pts(w, u=u, b=b, c=1)
    (x_lower, y_lower) = get_line_pts(w, u=u, b=b, c=-1)

    plt.plot(x_plane, y_plane, label='hyperplane', color='black')
    plt.plot(x_upper, y_upper, label='hyperplane', color='black', linestyle='dashed')
    plt.plot(x_lower, y_lower, label='hyperplane', color='black', linestyle='dashed')

    colors = np.where(labels == 1, 'red', 'blue')
    plt.scatter(x, y, c=colors, s=10, marker='x') # 'c' for colors, 's' for size

    # Set bounds of graph
    plt.ylim(-u, u)

    plt.show()