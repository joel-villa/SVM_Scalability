import numpy as np
import matplotlib.pyplot as plt
from generator import make_classification

class LinearSVC:
    """Linear SVC classifier.
    
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight 
      initialization.
    
    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    losses_ : list
      Number of misclassifications (updates) in each epoch.
    
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y, C=1):
        """Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of 
          examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        
        Returns
        -------
        self : object
        
        """
        n = y.size # number of samples

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []
        
        for _ in range(self.n_iter):
            # Adopted from perceptron loop
            y_hat = [] # The guesses of this itteration
            hinge_loss = []   
            for xi, yi in zip(X, y):
                y_hati = self.net_input(xi) # Guess for i'th sample in current itteration
                y_hat.append(y_hati)
                hinge_lossi = self._losses_(yi, y_hati) # i'th loss 
                hinge_loss.append()

                # Gradient
                grad_w = - self.eta * yi * xi
                grad_b = self.eta * yi

                self.w_ += grad_w
                self.b_ += grad_b
                # errors += update != 0.0
            ls = self._losses_(y, y_hat)
            loss = self._hinge_loss_(ls, C)
            self.losses_.append(loss)
        return self

    def fit_grad_descent(self, X, y, C=1):
        """Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of 
          examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        
        Returns
        -------
        self : object
        
        """
        n = y.size # number of samples

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []
        
        for i in range(self.n_iter):
            # Gradient Descent logic (ish)
            y_hat = self.net_input(X)
            # y_hat = self.predict(X)
            # errors = (y - net_input)
            loss = self._losses_(y, y_hat)

            #TODO: FIX THIS ahhhhhh, 

            # Gradient of the weights
            print(-np.dot(X.T, y))
            # print(loss)

            # print(y)
            grad_w = np.where((loss <= 0), 0, -np.dot(X.T, y))

            #Gradient of the bias 
            grad_b = y
            # print(f"loss: \n{loss}\ngrad_w: \n{grad_w}\n grad_b:\n{grad_b}")


            # self.w_ -= self.eta * grad_w
            self.b_ -= self.eta * grad_b

            hinge_loss = self._hinge_loss_(loss, C)
            self.losses_.append(hinge_loss)
        return self

    def fit_perceptron(self, X, y, C=1):
        """Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of 
          examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        
        Returns
        -------
        self : object
        
        """
        n = y.size # number of samples

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []
        
        for _ in range(self.n_iter):
            # Perceptron logic
            y_hat = []
            for xi, target, i in zip(X, y, range(n)):
                y_hat.append(self.net_input(xi))
                update = self.eta * (target - y_hat[i])
                self.w_ += update * xi
                self.b_ += update
                # errors += update != 0.0
            losses = self._losses_(y, y_hat)
            loss = self._hinge_loss_(losses, C)
            self.losses_.append(loss)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
    """
    Given the true labels y, and the labels generated by the linear SVC y_hat,
    Calculate the hinge loss of all samples
    """

    def _hinge_loss_(self, losses, C):
        weight_mag = np.linalg.norm(self.w_)
        return C * np.mean(losses) + 0.5 * weight_mag * weight_mag
    
    """
    The hinge lost function
    """
    def _losses_(self, y, y_hat):
        vals = 1 - y * y_hat
        # print(f"vals:{vals}")

        # If vals[i] is less than zero, put zero, otherwise keep vals[i]
        return np.where((vals <= 0 ), 0, vals)
        

if __name__ == "__main__":
    # i = 1
    X, y = make_classification(  10, rand_seed= 10)
    # print(y)
    lvc = LinearSVC(n_iter=50)
    lvc.fit(X, y)
    y_hat = lvc.predict(X)
    # print(y_hat)
    # print(lvc.losses_)

    fig, ax = plt.subplots(figsize=(16,8))

    ax.plot(range(1, len(lvc.losses_) + 1), (lvc.losses_), marker='')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.set_title("Linear SVC Losses over Epochs")
    plt.show()
