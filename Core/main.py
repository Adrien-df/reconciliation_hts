from reconciliation import To_Reconcile
import pandas as pd
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

if __name__ == "__main__":
    #input = input()
    print('\n Here is the output \n')

    data_m5 = pd.read_pickle("Data_Examples/M5_preprocessed.pkl")
    error_matrix = np.load("Data_Examples/error_matrix_m5_auto_arima.npy")
    #print(data_m5.head())
    print(data_m5['prediction'])

    summing = np.array([(1,1),(1,0),(0,1)])
    base = np.array([(3,2,2)]).T
    error = np.array([(1,0,1),(1,-1,0.5),(-1,0,1)])

    a=np.array([(1,2,3),(4,5,6),(1,1,3),(1,1,1),(1,0,0),(1,0,0),(0,1,0)])[:,2][:,None]
   
    #print(object.reconcile(method=input))
    #print(object.get_number_bottom_series())
    #print(object._check_parameters())
   
    object = To_Reconcile(data = data_m5, base_forecasts= data_m5['prediction'],columns_labels_ordered=['state_id','store_id', 'cat_id', 'dept_id'],in_sample_error_matrix=error_matrix)
    object.compute_summing_mat()
    print(object.reconcile(method='VS'))
    #print(object.score('rmse'))

    #print(data_m5['prediction'])
    
    #pour mes tests
    print('\n')

    