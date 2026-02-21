from LinearSVC import LinearSVC
from generator import make_classification
import numpy as np
import time
import matplotlib.pyplot as plt

ds = [10, 50, 100]
ns = [50, 5000, 50000]

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

def generate_random_datasets(seed_input=11, u_input=1):
    d10n500 = make_classification(n=500, d=10, u=u_input, rand_seed=seed_input)
    d10n5000 = make_classification(n=5000, d=10, u=u_input, rand_seed=seed_input)
    d10n50000 = make_classification(n=50000, d=10, u=u_input, rand_seed=seed_input)

    d50n500 = make_classification(n=500, d=50, u=u_input, rand_seed=seed_input)
    d50n5000 = make_classification(n=5000, d=50, u=u_input, rand_seed=seed_input)
    d50n50000 = make_classification(n=50000, d=50, u=u_input, rand_seed=seed_input)

    d100n500 = make_classification(n=500, d=100, u=u_input, rand_seed=seed_input)
    d100n5000 = make_classification(n=5000, d=100, u=u_input, rand_seed=seed_input)
    d100n50000 = make_classification(n=50000, d=100, u=u_input, rand_seed=seed_input)

    return [d10n500, d10n5000, d10n50000, 
            d50n500, d50n5000, d50n50000, 
            d100n500, d100n5000, d100n50000]

def test_svc_loss_convergence_and_time(datasets, C=15, eta=0.01, n_iter=100):
    idx_count = 0
    results = {}

    for d in ds:
        for n in ns:
            X, y, _, _, _ = datasets[idx_count]

            lvc = LinearSVC(eta=eta, n_iter=n_iter)
            start = time.perf_counter()
            lvc.fit(X=X, y=y, C=C)
            elapsed = time.perf_counter() - start

            results[(d, n)] = {
                "losses": lvc.losses_,
                "fit_time": elapsed
            }

            idx_count += 1

    return results

def plot_loss_grid(results):
    fig, axs = plt.subplots(len(ds), len(ns), figsize=(18, 12), sharey=True)
    fig.suptitle("Loss Convergence across datasets", fontsize=16)

    fig.subplots_adjust(
        top=0.90,
        hspace=0.5,
        wspace=0.3
    )

    for i, d in enumerate(ds):
        for j, n in enumerate(ns):
            ax = axs[i][j]
            info = results[(d, n)]
            losses = info["losses"]
            fit_time = info["fit_time"]

            ax.plot(range(1, len(losses) + 1), losses)
            ax.set_title(f"d={d}, n={n}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

            print(f"d: {d}, n: {n}, time: {fit_time}s")

            ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("plots/loss_convergence_grid.svg")
    plt.show()

if __name__ == "__main__":
    # test_C_values()
    datasets = generate_random_datasets(seed_input=11, u_input=1)
    results = test_svc_loss_convergence_and_time(
        datasets=datasets,
        C=15,
        eta=0.01,
        n_iter=100
    )
    plot_loss_grid(results=results)

    # print(datasets[0][0])