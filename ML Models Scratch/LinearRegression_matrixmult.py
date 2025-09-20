import numpy as np
import matplotlib.pyplot as plt
class LinearRegression:
    def __init__(self,alpha = 0.03, max_iter = 100, theta_init = None, tol = 1e-4, lambd = 0, penalty = None, theta_history = None, cost_history = None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta_init = theta_init
        self.tol= tol
        self.lambd = lambd
        self.penalty = penalty
        self.theta_history = theta_history
        self.cost_history = cost_history
    
    def compute_cost(self,theta,X,y):
        #X 200,3
        #theta = 3,1
        #theta
        if self.penalty == None:
            #MSE = 1/N * (np.sum(theta@X - y)**2)
            mse_cost = float(1/(X.shape[0]) * np.sum((X@theta - y)**2))
            return mse_cost
    
        if self.penalty == 'l1':
            mse_cost = float(1/(X.shape[0]) * np.sum((X@theta - y)**2))
            cost = float(mse_cost + self.lambd * np.linalg.norm(theta[1:],1))
            return cost
        
        if self.penalty == 'l2':
            mse_cost = float(1/(X.shape[0]) * np.sum((X@theta - y)**2))
            cost = float(mse_cost + self.lambd * np.linalg.norm(theta[1:],2))
            return cost
        
    def compute_gradient(self,X,y, theta):
        if self.penalty is None:
            #grad = -2/N * XT @ (y-X@theta)
            residual = y - X@theta
            grad = -2/X.shape[0] * (X.T @ residual)
            #print("grad shape" , grad.shape)
            return grad
            
        if self.penalty == 'l1':
            residual = y - X@theta
            grad_linear = -2/X.shape[0] * (X.T @ residual)
            reg =  np.sign(theta[1:])
            reg[0] = 0
            grad = grad_linear + reg
            return grad
        
        if self.penalty == 'l2':
            residual = y - X@theta
            grad_linear = -2/X.shape[0] * (X.T @ residual)
            reg =  2*np.sign(theta[1:])
            reg[0] = 0
            grad = grad_linear + reg
            return grad
    def has_converged(self, theta,theta_old ):
        if np.sum(np.abs(theta.T)) - np.sum(np.abs(theta_old.T)) < self.tol:
            return True
        else:
            return False
        
    def fit(self, X,y):
        N,d = X.shape
        #1, Add intercept term to X
        ones = np.ones((N,1))
        X = np.hstack((ones,X)) #X = 200,3
        
        #theta - [1,3]
        if self.theta_init is None:
            theta_init = np.zeros((d+1,1))
        
        cost_init = self.compute_cost(theta_init,X,y)   
        
        self.theta_history = [theta_init.copy()]
        self.cost_history = [cost_init]     
        
        theta = theta_init.copy()
        for i in range(0,self.max_iter+1):
            
            #calculate gradients
            grad = self.compute_gradient(X,y, theta)
            #update theta
            theta_old = theta.copy()
            theta = theta_old - self.alpha*grad
            self.theta_history.append(theta)
            self.cost_history.append(self.compute_cost(theta,X,y))
            
            if self.has_converged(theta, theta_old):
                return
            
    def predict(self,X):
        #add intercept
        ones = np.ones((X.shape[0],1)) #20,1
        X = np.hstack((ones,X)) #20,3
        
        #theta = 1,3
        theta = self.theta_history[-1]
        y_pred = X@theta
        return y_pred
    
    def calculate_r2(self, y_pred,y_test):
        N = y_test.shape[0]
        y_pred = y_pred.ravel()
        y_test = y_test.ravel()
        rss = np.sum((y_pred - y_test)**2)
        tss = np.sum((y_pred -y_test.mean() )**2)
        if tss ==0:
            return tss
        r2 = 1- (rss/tss)
        return r2
        
    
def main():
    #X = 100,2
    #noise = 100,1
    #y = 100,1
    X = np.random.randn(100,2)
    noise = np. random.randn(100,1)
    y = 5 + 10*X[:,[0]] + 20*X[:,[1]]
    
    ind = np.random.permutation(X.shape[0])
    ind_train = ind[:80]
    ind_test = ind[80:]
    X_train,X_test,y_train,y_test = X[ind_train],X[ind_test],y[ind_train,],y[ind_test]
    #print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)    
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    print(y_pred.shape,y_test.shape)
    r2_score = lr.calculate_r2(y_pred,y_test)
    print(r2_score)
    return r2_score
    
    
    
if __name__ == "__main__":
    main()