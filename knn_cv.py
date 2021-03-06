import numpy as np
from scipy.stats import rankdata
from sklearn import datasets
from matplotlib import pyplot as plt

def knn_cv(data, fold=5, K=10):
    print("Fold = " + str(fold))
    max_er = 1                          
    
    #K different values for k to find the k with the least smalles error rate
    for k in range(1,K+1):
        set_len = int(data.shape[0] / fold)
        fehler = 0
        for i in range(set_len,data.shape[0]+1, set_len):
            #Creating a test and training sample from the original data set
            test_data = data[i-set_len:i]
            training_data = np.delete(data, np.s_[i-set_len:i],0)
            
            #For each datapoint in the test sample
            for j in range(set_len):
                #Calculating the Frobenius Norm to each training observation
                distance = np.sum((training_data[:,:-1] - test_data[j,:-1])**2, axis=1)
                
                #Ranks the Frobenius Norms to find the k-smallest distances
                rank = rankdata(distance, method="ordinal")
                
                n_classes = np.array([])
                for k_2 in range(k):
                    for l in range(rank.shape[0]):
                        if rank[l] == k_2+1:
                            n_classes = np.append(n_classes, training_data[l,-1])
                
                #Determine the most frequent class within the KNN
                count = np.bincount(np.array(n_classes, dtype=int))
                
                max = -1
                for m in range(count.shape[0]):
                    if count[m] > max:
                        max = count[m]
                        decision = m
                
                #Error Rate
                if decision != test_data[j, -1]:
                    fehler += 1
        
        d_fehler = fehler / data.shape[0]
        if d_fehler < max_er:
            rec_k = k
            max_er = d_fehler
        
        print("Error Rate for " + str(k) + "-NN: " + str(round(d_fehler*100,1)) + " %")
    print("Recommended K: " + str(rec_k))
        
        
if __name__ == '__main__':
    #Example:
    iris = datasets.load_iris()
    data = iris["data"][:,:2]
    klasse = iris["target"]

    data = np.concatenate((data, klasse.reshape(klasse.shape[0],1)), axis=1) 

    knn_cv(data, fold=5)