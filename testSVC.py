from sklearn.svm import LinearSVC
from LinearSVC import LinearSVC as ImplementedSVC
from generator import make_classification
import numpy as np
import time
import matplotlib.pyplot as plt

N_ITER = 100
STEP = N_ITER // 10
TOLERANCE = 0.1
C = 15


ds = [10, 50, 100]
ns = [500, 5000, 50000]

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

def test_svc_loss_convergence_and_time(datasets):
    idx_count = 0
    results = {}

    for d in ds:
        for n in ns:
            X, y, _, _, _ = datasets[idx_count]

            lvc = ImplementedSVC(eta=TOLERANCE, C=C, n_iter=N_ITER)
            start = time.perf_counter()
            lvc.fit(X=X, y=y)
            elapsed = time.perf_counter() - start

            results[(d, n)] = {
                "losses": lvc.losses_,
                "fit_time": elapsed
            }

            idx_count += 1

    return results

def hinge_loss(y, y_hat, w, C):
    vals = 1 - y * y_hat

    # If vals[i] is less than zero, put zero, otherwise keep vals[i]
    loss = np.where((vals <= 0 ), 0, vals)

    # weight_mag = np.linalg.norm(w)

    return np.mean(loss) 
    # return C * np.mean(loss) #+ 0.5 * weight_mag * weight_mag


def test_svc_loss_convergence_and_time_scikit(type, datasets):
    idx_count = 0
    results = {}

    for d in ds:
        for n in ns:
            X, y, _, _, _ = datasets[idx_count]

            lvc = LinearSVC(dual=type, tol=TOLERANCE, C=C, max_iter=N_ITER)
            start = time.perf_counter()
            lvc.fit(X=X, y=y)
            elapsed = time.perf_counter() - start
            
            losses = []
            for i in range(1,N_ITER, STEP):
                lvc = LinearSVC(dual=type, tol=TOLERANCE, C=C, max_iter=i)
                lvc.fit(X=X, y=y)

                y_hat = lvc.predict(X)
                loss = hinge_loss(y=y, y_hat=y_hat, w=lvc.coef_, C=C)
                losses.append(loss)

            results[(d, n)] = {
                "losses": losses,
                "fit_time": elapsed
            }

            idx_count += 1

    return results

def plot_loss_grid(results, type):
    fig, axs = plt.subplots(
        len(ds),
        len(ns),
        figsize=(20, 14),
        sharey=True
    )

    for i, d in enumerate(ds):
        for j, n in enumerate(ns):
            ax = axs[i][j]
            info = results[(d, n)]
            losses = info["losses"]
            fit_time = info["fit_time"]

            ax.plot(range(1, len(losses) + 1), losses)

            ax.set_title(
                f"d={d}, n={n}",
                fontsize=16,
                pad=10
            )

            ax.set_xlabel("Epoch", fontsize=14, labelpad=6)
            ax.set_ylabel("Loss", fontsize=14, labelpad=6)

            ax.tick_params(axis='both', labelsize=12)

            ax.grid(True)

            print(f"d: {d}, n: {n}, time: {fit_time}s")

    fig.subplots_adjust(
        top=0.92,
        bottom=0.08,
        left=0.08,
        right=0.98,
        hspace=0.6,
        wspace=0.4
    )

    plt.savefig("plots/loss_convergence_grid" + type + ".svg")
    plt.show()

if __name__ == "__main__":

    # test_C_values()
    datasets = generate_random_datasets(seed_input=11, u_input=1)

    # OUR IMPLEMENTATION
    results = test_svc_loss_convergence_and_time(
        datasets=datasets,
    )
    print("IMPLEMENTED")
    plot_loss_grid(results=results, type="Implemented")


    # PRIMAL 
    results = test_svc_loss_convergence_and_time_scikit(
        datasets=datasets,
        type=False #dual = false
    )
    print("PRIMAL")
    plot_loss_grid(results=results, type="Primal")

     # DUAL 
    results = test_svc_loss_convergence_and_time_scikit(
        datasets=datasets,
        type=True #dual = true
    )
    print("DUAL")
    plot_loss_grid(results=results, type="Dual")

    # print(datasets[0][0])