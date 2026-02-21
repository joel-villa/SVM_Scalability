from sklearn.svm import LinearSVC
from LinearSVC import LinearSVC as ImplementedSVC

N_ITER = 1000
TOLERANCE = 0.0001
C = 15


if __name__ == "__main__":
    svc_dual   = LinearSVC(dual=True,  tol=TOLERANCE, C=C, max_iter=N_ITER, loss='hinge')
    svc_primal = LinearSVC(dual=False, tol=TOLERANCE, C=C, max_iter=N_ITER, loss='hinge')
    