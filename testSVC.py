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
    
def get_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_true == y_pred)

def test_C_values():
    Cs = [1, 5, 10, 15, 20, 50, 100, 200, 500, 1000]
    X, y, a = make_classification(100, rand_seed= 10)
    for C_val in Cs:
        lvc = LinearSVC(n_iter=1000, eta=0.01)
        lvc.fit(X, y, C=C_val)
        y_hat = lvc.predict(X)
        acc = get_accuracy(y, y_hat)

        print(f"C: {C_val}, Training accuracy: {acc:.4f}")

def generate_random_datasets():
    pass

if __name__ == "__main__":
    test_C_values()