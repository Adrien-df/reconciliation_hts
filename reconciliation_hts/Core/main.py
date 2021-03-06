from reconciliation import To_Reconcile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    data_m5 = pd.read_pickle(
        "reconciliation_hts/Data_Examples/M5_preprocessed.pkl")
    error_matrix = np.load(
        "reconciliation_hts/Data_Examples/error_matrix_500_to_600.npy",
        allow_pickle=True)

    #predictions = np.load("Data_Examples/predictions_auto_arima_m5_500_to_600.npy",allow_pickle=True)
    #reals = np.load("Data_Examples/real_value_m5_600_to_500.npy",allow_pickle=True)

    predictions = np.load(
        "reconciliation_hts/Data_Examples/predictions_auto_arima_m5_1000_to_1100.npy",
        allow_pickle=True)
    reals = np.load(
        "reconciliation_hts/Data_Examples/real_value_m5_1000_to_1100.npy",
        allow_pickle=True)

    object = To_Reconcile(data=data_m5, base_forecasts=predictions,
                          columns_ordered=[
                              'state_id', 'store_id', 'cat_id', 'dept_id'],
                          error_matrix=error_matrix, real_values=reals)

    #print(object.reconcile(method='MinTSh', reconcile_all=True))
    # object.score()

    #print(object.cross_score(metrics='rmse',reconcile_method='MinTSh',test_all=True))

    object.plot(level='state_id', reconcile_method='MinTSh', plot_real=True)
    # print(object.proba_reconcile(column_to_reconcile=1))

    print('\n')
