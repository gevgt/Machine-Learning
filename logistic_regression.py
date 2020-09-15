import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets


def logit(reg_x, reg_y):
    
    ############################################################################
    ### Scatterplot der Beobachtungen: Klasse und erste erklärenden Variable ###
    ############################################################################
    row, col = reg_x[:,1:].shape
    
    if col == 1:
        null = np.where(reg_y==0, reg_x[:,1].reshape(row,1), np.nan)
        eins = np.where(reg_y==1, reg_x[:,1].reshape(row,1), np.nan)
        
        plt.figure()
        plt.xlabel("x_1")
        plt.ylabel("p(x)")
        plt.title("Logistic Regression:\nProbabilities for different Values of x_1")
        plt.grid(color='grey', linestyle='-', linewidth=0.1)
        plt.scatter(null, np.zeros((row)), color="blue")
        plt.scatter(eins, np.ones((row)), color="red")
    
    
    
    
    
    ###################################
    ### Fitting Logistic Regression ###
    ###################################
    
    #Starting points for optimization
    beta = np.matmul(np.linalg.inv(np.matmul(reg_x.T,reg_x)), np.matmul(reg_x.T, reg_y))
    
    
    #Numeric solution: Newton Raphson Method
    for i in range(100):
        y_hat = np.array([float((1 + np.exp(-np.matmul(beta.T,i)))**(-1)) for i in reg_x], 
                          dtype=float).reshape(row, 1)
        W = np.diag(y_hat.reshape(row,))
        
        #First and second derivative
        grad_b = np.matmul(reg_x.T, reg_y-y_hat)
        grad_bb = - np.matmul(np.matmul(reg_x.T, W), reg_x)
        
        #Newton Raphson equation
        beta = beta - np.matmul(np.linalg.inv(grad_bb), grad_b)
    
    
    #Maximum Likelihood    
    l_b = 0
    for i in range(reg_x.shape[0]):
        l_b += reg_y[i] * np.matmul(beta.T, reg_x[i]) - np.log(1 + 
                    np.exp(np.matmul(beta.T, reg_x[i])))
    
    
    #Output Table
    idx = []
    for i in range(col+1):
        if i == 0:
            idx.append("Intercept")
        else:
            idx.append("β_"+str(i))
    
    beta = pd.DataFrame(beta, index=idx, columns=["Regressors"])
    
    print("Maximum Likelihood: " + str(float(np.round(l_b,4))) + "\n")
    print(beta)

    
    
    
    #####################################################
    ### Grafik der "fitted logistic regression curve" ###
    #####################################################
    
    if col == 1:
        start = np.min(reg_x[:,1])
        end = np.max(reg_x[:,1])
        n = 150
        
        for o in np.linspace(start, end, n):
            x_grafik = [1, o]
            plt.scatter(o, (1 + np.exp(- np.matmul(beta.values.T, x_grafik)))**-1, color="black", s=7)

        plt.legend(["Class 0", "Class 1", "Prob. for x"], fontsize="x-small")
        plt.show()
            



if __name__ == '__main__':
    iris = datasets.load_iris()
    reg_x = np.concatenate([np.ones((100,1)), iris["data"][:100,0].reshape(100,1)], axis=1)
    reg_y = iris["target"][:100].reshape(100,1)
     
    logit(reg_x, reg_y)
