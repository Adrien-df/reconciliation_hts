from reconciliation import To_Reconcile
import pandas as pd
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

if __name__ == "__main__":
    #input = input()

    data_m5 = pd.read_pickle("Data_Examples/M5_preprocessed.pkl")
    error_matrix = np.load("Data_Examples/error_matrix_500_to_600.npy",allow_pickle=True)
    predictions = np.load("Data_Examples/predictions_auto_arima_m5_500_to_600.npy",allow_pickle=True)
    reals = np.load("Data_Examples/real_value_m5_600_to_500.npy",allow_pickle=True)
    #print(data_m5.head())
    #print(predictions[:,0])

   
    object = To_Reconcile(data = data_m5, base_forecasts= predictions,columns_labels_ordered=['state_id','store_id', 'cat_id', 'dept_id'],in_sample_error_matrix=error_matrix,real_values=reals)
    object.compute_summing_mat()
    #object.reconcile(method='MinTSh')
    print(object.cross_val_score(metrics='mase',reconcile_method='MinTSh',test_all=True))
    object.plot(level='total',reconcile_method='MinTSh')

    #print(data_m5['prediction'])
    
    print('\n')

    