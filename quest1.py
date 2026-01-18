import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data = sio.loadmat('lsdata.mat')
    X = data['X']
    Y = data['Y']
    X_test = data['Xtest']
    Y_test = data['Ytest']
    return X, Y, X_test, Y_test

def least_squares(X, Y):
    """
    Computes the weight vector w for the Least Squares problem.
    w = (X^T X)^-1 X^T Y
    """
    w = np.linalg.inv(X.T @ X) @ X.T @ Y
    return w

def ridge_regression(X, Y, lmbda):
    """
    Computes the weight vector w for the Ridge Regression problem.
    w = (X^T X + lambda * I)^-1 X^T Y
    """
    d = X.shape[1]
    w = np.linalg.inv(X.T @ X + lmbda * np.eye(d)) @ X.T @ Y
    return w

def squared_loss(X, Y, w):
    m = X.shape[0]
    predictions = X @ w
    loss = np.sum((predictions - Y)**2) / m
    return loss

def run_experiment_1a():
    X_train_full, Y_train_full, X_test, Y_test = load_data()
    
    m_values = list(range(100, 510, 10))
    train_losses = []
    test_losses = []
    
    for m in m_values:
        # Sample m points from the training set
        # The prompt says "sample m points", usually this means random sampling 
        # or just taking the first m points if they are already shuffled.
        # To be safe and consistent, I'll take the first m points or random?
        # Usually "sample" implies random.
        indices = np.random.choice(X_train_full.shape[0], m, replace=False)
        X_train = X_train_full[indices]
        Y_train = Y_train_full[indices]
        
        w = least_squares(X_train, Y_train)
        
        train_loss = squared_loss(X_train, Y_train, w)
        test_loss = squared_loss(X_test, Y_test, w)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
    # Plotting
    plt.figure(figsize=(10, 5))
    
    # i. Test loss plot
    plt.subplot(1, 2, 1)
    plt.plot(m_values, test_losses, marker='o')
    plt.title('Average Squared Loss on Test Set')
    plt.xlabel('Training Set Size (m)')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # ii. Train loss plot
    plt.subplot(1, 2, 2)
    plt.plot(m_values, train_losses, marker='o', color='orange')
    plt.title('Average Squared Loss on Training Set')
    plt.xlabel('Training Set Size (m)')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def run_experiment_1b(m, lmbda_values):
    X_train_full, Y_train_full, X_test, Y_test = load_data()
    
    # Sample m points
    indices = np.random.choice(X_train_full.shape[0], m, replace=False)
    X_train = X_train_full[indices]
    Y_train = Y_train_full[indices]
    
    # Least Square solution for baseline
    w_ls = least_squares(X_train, Y_train)
    ls_test_loss = squared_loss(X_test, Y_test, w_ls)
    
    ridge_test_losses = []
    for lmbda in lmbda_values:
        w_ridge = ridge_regression(X_train, Y_train, lmbda)
        test_loss = squared_loss(X_test, Y_test, w_ridge)
        ridge_test_losses.append(test_loss)
        
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(lmbda_values, ridge_test_losses, marker='o', label='Ridge Regression')
    plt.axhline(y=ls_test_loss, color='r', linestyle='--', label='Least Squares Baseline')
    plt.title(f'Test Loss vs Lambda (m={m})')
    plt.xlabel('Lambda')
    plt.ylabel('Average Squared Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    X, Y, Xt, Yt = load_data()
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"X_test shape: {Xt.shape}")
    print(f"Y_test shape: {Yt.shape}")
    
    print("\nRunning Experiment 1a...")
    run_experiment_1a()
    
    lmbda_values = [0, 0.01, 0.02, 0.05, 0.1, 1, 10, 15]
    
    print("\nRunning Experiment 1b (m=60)...")
    run_experiment_1b(60, lmbda_values)
    
    print("\nRunning Experiment 1b (m=500)...")
    run_experiment_1b(500, lmbda_values)