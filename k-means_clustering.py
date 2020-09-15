import numpy as np
from scipy.stats import rankdata
from sklearn import datasets
from matplotlib import pyplot as plt


    
def clustering(data, n_groups, laps=10):
    d_sum_max = np.max(data) * 10000
    
    for lap in range(laps):
        #Random allocation of groups
        group = np.random.randint(n_groups, size = data.shape[0])
        
        #Makes sure to pass the first reallocation process
        first = True
        
        while True:    
            #Skips this part in the first loop, since a mean of the groups is required
            if first == False:
                
                #Reallocates the observations to the group to which mean they are closest to
                comp = np.copy(group)
                for i in range(data.shape[0]):
                    distance = np.max(data) * 10000
                    for j in range(n_groups):     
                        if np.sum((data[i]-mean[j])**2) < distance:
                            distance = np.sum((data[i]-mean[j])**2)
                            group[i] = j    
            
            #Calculation of the groups mean    
            mean = np.ones((1,data.shape[1]))
            for i in range(n_groups):
                v_summe = np.ones((1,data.shape[1]))
                n = 0
                for j in range(data.shape[0]):
                    if group[j] == i:
                        v_summe = v_summe + data[j]
                        n += 1
                v_mean = v_summe / n
                mean = np.concatenate((mean, v_mean.reshape(1,data.shape[1])), axis=0)
            mean = mean[1:]
            
            #Breaks the loop if the observations do not change the groups anymore
            if first == False:    
                if np.sum(comp-group) == 0:
                    break
                
            #Enables the reallocation after the first loop
            first = False
        
        #Aggregated sum of distance between observation and groups mean
        d_sum = 0
        for i in range(data.shape[0]):
            d_sum += np.sum((data[i]-mean[group[i]])**2)
        
        #Saves the group allocation with the smallest aggregated distance
        if d_sum < d_sum_max:
            d_sum_max = d_sum
            group_final = np.copy(group)
            mean_final = np.copy(mean)
    
    #Plots a chart if the dataset is 2D
    if data.shape[1] == 2:
        plt.figure()
        for i in range(n_groups):
            plot_data = np.ones((1, data.shape[1]))
            for j in range(group_final.shape[0]):
                if group_final[j] == i:
                    plot_data = np.concatenate((plot_data, data[j].reshape(1,2)), axis=0)
            plot_data = plot_data[1:]
            plt.scatter(plot_data[:,0], plot_data[:,1])
        
        for i in range(n_groups):
            plt.scatter(mean_final[i,0], mean_final[i,1], linewidth=3, color="red")
        plt.show()
        
    return group_final, d_sum
        
        
if __name__ == '__main__':
    #Iris Example:       
    iris = datasets.load_iris()
    data = iris["data"][:,:2]

    clustering(data, n_groups=3)
