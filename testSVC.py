from LinearSVC import LinearSVC
from generator import make_classification
import numpy as np

def test(d_max):
    n = d_max * d_max
    for d in range(d_max)[::-10]:
        X, y = make_classification(n, d, rand_seed=d)
        lvc = LinearSVC()
        lvc.fit(X, y)
        y_hat = lvc.predict(X)

        num_diff = np.sum(y != y_hat)
        if (num_diff > 0):
            print(f"n = {n}, d = {d}, num_diff = {num_diff}")
            # print(f"y: {y}")
            # print(f"y_hat: {y_hat}")
    

if __name__ == "__main__":
    test(100)
    # # i = 1
    # X, y = make_classification(  10, rand_seed= 10)
    # print(y)
    # lvc = LinearSVC()
    # lvc.fit(X, y)
    # y_hat = lvc.predict(X)
    # print(y_hat)