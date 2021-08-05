from typing import Optional, Union
import numpy as np 
import pandas as pd
from typing import Optional, Union, Iterable, Tuple, List
from numpy.typing import ArrayLike
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import random
import utils
import matplotlib.pyplot as plt
from statsmodels.stats.moment_helpers import cov2corr
import warnings


class To_Reconcile:

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        columns_ordered: Optional[list[str]] = None,
        summing_mat: Optional[np.ndarray] = None,
        base_forecasts: Optional[np.ndarray] = None,
        real_values: Optional[np.ndarray] = None,
        in_sample_error_matrix: Optional[np.ndarray] = None,
        lambd=None,
        inputs_are_checked=False
    ) -> None:
        self.data = data
        self.summing_mat = summing_mat
        self.in_sample_error_matrix = in_sample_error_matrix
        self.base_forecasts = base_forecasts
        self.columns_ordered = columns_ordered
        self.real_values = real_values
        self.lambd = lambd
        self.inputs_are_checked = inputs_are_checked

    def _check_compatibility(self) -> None:
        """
        [Performs several checks on the compatibility of the inputs provided]

        [extended_summary]

        Raises
        ------
        ValueError
            [If mutually exclusive parameters are given in input]
        """
        if not self.inputs_are_checked:

            if self.data is not None:
                if self.summing_mat is not None:
                    raise ValueError(
                        "data and summing_mat were provided."
                        "You provide summing_mat OR data + columns_ordered"
                    )

            if self.columns_ordered is not None:
                if self.summing_mat is not None:
                    raise ValueError(
                        "columns_ordered and summing_mat were provided."
                        "You provide summing_mat OR data + columns_ordered "
                    )

            if self.data is None and self.summing_mat is None:
                raise ValueError(
                    "You provide summing_mat OR data + columns_ordered"
                )

    def _check_parameters(self) -> None:
        """
        [Perform several checks on input parameters]

        [extended_summary]

        Raises
        ------
        ValueError
            [If one of the parameters is not valid]  
        """
        if not self.inputs_are_checked:

            if not isinstance(self.data, pd.DataFrame):
                raise ValueError(
                    "Invalid type for data."
                    "Expected pd.DataFrame object"
                )
            if self.summing_mat is not None:
                if not isinstance(self.summing_mat, np.ndarray):
                    raise ValueError(
                        "Invalid type for summing matrix"
                        " Expected np.ndarray type"
                    )
                if self.summing_mat.shape[0] < 3:
                    raise ValueError(
                        "Invalid summing_mat shape. Must be of shape >= 3."
                    )            
            if not isinstance(self.base_forecasts, np.ndarray):
                raise ValueError(
                    "Invalid type for base forecasts. Expected np.ndarray type"
                )

            if not isinstance(self.real_values, np.ndarray):
                raise ValueError(
                    "Invalid type for real values. Expected numpy array type"
                )

            if not isinstance(self.in_sample_error_matrix, np.ndarray):
                raise ValueError(
                    "Invalid type for error_matrix. Expected numpy array type"
                )

            if self.data is not None:

                for element in self.columns_ordered:
                    check = 1
                    if element not in self.data.columns:
                        check = 0
                        break
                if check == 0:
                    raise ValueError(
                        "Columns name provided in columns_ordered not in data"
                        "Check the columns and their names"
                        )

                if self.data.shape[0] != self.base_forecasts.shape[0]:
                    raise ValueError(
                        f" data has {self.data.shape[0]} rows"
                        "base_forecast has {self.base_forecasts.shape[0]} rows"
                        "Same value is expected"
                    )

                if self.real_values is not None:
                    if self.data.shape[0] != self.real_values.shape[0]:
                        raise ValueError(
                            f" data has {self.data.shape[0]} rows"
                            "real_values has {self.real_values.shape[0]} rows"
                            "Same value is expected."
                        )
                if self.in_sample_error_matrix is not None:
                    n = self.data.shape[0]
                    m = self.in_sample_error_matrix.shape[0]
                    if n != m:
                        raise ValueError(
                            f" data has {self.data.shape[0]} rows"
                            "in_sample_error_matrix has"
                            f"{self.in_sample_error_matrix.shape[0]} rows"
                            "Same value is expected."
                        )

    def compute_summing_mat(
        self
    ) -> np.ndarray:
        """
        [Computes the summing_matrix]

        [When data and columns_label_ordered is passed in input,
        this method will automatically compute the summing matrix]

        Returns
        -------
        np.ndarray
            [The summing matrix used for reconciliation afterwards]
        """

        L = len(self.columns_ordered)  # assert that the lenght is sufficient
        n = self.data.shape[0]
        m = self.data[self.columns_ordered].isna().any(axis=1).sum()
        self.summing_mat = np.zeros([n, n-m])
        self.summing_mat[0] = np.ones(n-m)
        self.summing_mat[m:] = np.identity(n-m)

        for level in range(1, L):

            for i in range(1, n-m):
                if pd.isna(self.data.iloc[i][self.columns_ordered[level]]) and not pd.isna(self.data.iloc[i][self.columns_ordered[level-1]]):
                    list_of_values = [self.data.iloc[i][self.columns_ordered[k]] for k in range(level)]

                    for j in range(m, n):
                        if [self.data.iloc[j][self.columns_ordered[k]] for k in range(level)] == list_of_values:
                            self.summing_mat[i][j-m] = 1
        return(self.summing_mat)

    def _get_indexes_level(
        self
    ) -> dict:
        dictionnary = {'total': 0}
        L = len(self.columns_ordered)  # assert that the lenght is sufficient
        n = self.data.shape[0]

        for level in range(1, L):
            indexes_of_level = []
            for i in range(1, n):
                if pd.isna(self.data.iloc[i][self.columns_ordered[level]]) and not pd.isna(self.data.iloc[i][self.columns_ordered[level-1]]):
                    indexes_of_level.append(i)
            dictionnary[self.columns_ordered[level-1]] = indexes_of_level
        indexes_of_level = []
        for i in range(1, n):
            if not pd.isna(self.data.iloc[i][self.columns_ordered[-1]]):
                indexes_of_level.append(i)
        dictionnary[self.columns_ordered[-1]] = indexes_of_level
        return(dictionnary)

    def get_number_bottom_series(
        self,
    ) -> int:
        n = (self.summing_mat).shape[0]
        i = 0
        while self.summing_mat[i].sum() != 1:
            i += 1
        return (n-i)

    def _compute_lambda(
        self
    ) -> float:
        res = np.matrix(self.in_sample_error_matrix).T
        number_error_vectors = res.shape[0]
        covm = utils.cross_product(res)/number_error_vectors
        corm = cov2corr(covm)
        xs = utils.scale(res, np.sqrt(np.diag(covm)))
        v = (1/(number_error_vectors * (number_error_vectors - 1))) * (utils.cross_product(np.square(np.matrix(xs))) - 1/number_error_vectors * (np.square(utils.cross_product(np.matrix(xs)))))
        np.fill_diagonal(v, 0)
        corapn = cov2corr(np.diag(np.diag(covm)))
        d = np.square((corm - corapn))
        lambd = v.sum()/d.sum()
        lambd = max(min(lambd, 1), 0)
        self.lambd = lambd
        return(lambd)

    def reconcile(
        self,
        method: Optional[str] = 'MintSa',
        column_to_reconcile: Optional[int] = -1,
        reconcile_all: Optional[bool] = False,
        _vector_to_proba_reconcile: Optional[np.ndarray] = None,
        show_lambda: Optional[bool] = False
    ) -> ArrayLike:
        """[summary]

        [extended_summary]

        Parameters
        ----------
        method : Optional[str], optional
            [description], by default 'MintSa'
        column_to_reconcile : Optional[int], optional
            [description], by default -1
        reconcile_all : Optional[bool], optional
            [description], by default False

        Returns
        -------
        ArrayLike
            [description]

        Raises
        ------
        ValueError
            [description]
        """
        self._check_compatibility()
        self._check_parameters()
        self.inputs_are_checked = True

        if self.summing_mat is None:
            self.summing_mat = self.compute_summing_mat()

        if method == 'OLS':
            combination_matrix = np.linalg.inv(np.transpose(self.summing_mat)@self.summing_mat)@np.transpose(self.summing_mat)

        elif method == 'BU':
            combination_matrix = np.concatenate((np.zeros(shape=(self.get_number_bottom_series(), self.summing_mat.shape[0]-self.get_number_bottom_series())), np.identity(self.get_number_bottom_series())), axis=1)

        elif method == 'SS':

            W = np.diag(self.summing_mat@np.ones(self.get_number_bottom_series()))
            combination_matrix = np.linalg.inv((self.summing_mat.T)@np.linalg.inv(W)@self.summing_mat)@(self.summing_mat.T)@np.linalg.inv(W)

        elif method in ['VS', 'MinTSa', 'MinTSh']:

            number_error_vectors = self.in_sample_error_matrix.shape[1]
            W1 = np.zeros((self.summing_mat.shape[0], self.summing_mat.shape[0]))

            for i in range(number_error_vectors):
                W1 += (self.in_sample_error_matrix[:, i][:, None])@(self.in_sample_error_matrix[:, i][:, None].T)
            W1 = W1/number_error_vectors

            if method == 'VS':
                W = np.diag(np.diag(W1))

            elif method == 'MinTSa':
                W = W1

            elif method == 'MinTSh':
                if self.lambd is None:
                    lambd = self._compute_lambda()
                    if show_lambda:
                        print("lambda parameter for MinTSh reconciliation method is equal to : ", lambd)
                else: 
                    lambd = self.lambd

                W = lambd*np.diag(np.diag(W1)) + (1-lambd) * W1

            combination_matrix = np.linalg.inv((self.summing_mat.T)@np.linalg.inv(W)@self.summing_mat)@(self.summing_mat.T)@np.linalg.inv(W)

        else:
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'OLS','BU', SS', 'VS', 'MinTSa' and 'MinTSh'"
            )

        if _vector_to_proba_reconcile is not None:
            return(self.summing_mat@combination_matrix@_vector_to_proba_reconcile)

        elif reconcile_all:
            return( [self.summing_mat@combination_matrix@self.base_forecasts[:, i] for i in len(self.base_forecasts)])

        elif column_to_reconcile == -1:
            return(self.summing_mat@combination_matrix@self.base_forecasts)

        else:
            return(self.summing_mat@combination_matrix@self.base_forecasts[:,column_to_reconcile])




    def score(
        self,
        metrics: Optional[str] = 'rmse',
        reconcile_method: Optional[str] = 'MinTSa',

    ) -> pd.DataFrame:

    #check that there is only one vector in base forecasts

        if metrics == 'rmse':
            score_dataframe = pd.DataFrame(data ={'rmse': [mean_squared_error(self.real_values, self.base_forecasts, squared=False),
            mean_squared_error(self.real_values, self.reconcile(method=reconcile_method), squared=False)]})

        elif metrics ==' mse':
            score_dataframe = pd.DataFrame(data ={'rmse': [mean_squared_error(self.real_values, self.base_forecasts, squared=False),
            mean_squared_error(self.real_values, self.reconcile(method=reconcile_method), squared=False)]})

        elif metrics == 'mase':
            score_dataframe = pd.DataFrame(data ={'mase': [mean_absolute_error(self.real_values, self.base_forecasts),
            mean_absolute_error(self.real_values, self.reconcile(method=reconcile_method))]})

        score_dataframe.rename(index={0: "Base forecast", 1: f"Reconciled forecast ({reconcile_method})"}, inplace=True)

        return(score_dataframe)


    def cross_val_score( #You have to make sure that the shape of real values and predictions are the same and of size n 
        self,
        reconcile_method: Optional[str] = 'MinTSa',
        metrics: Optional[str] = 'rmse',        
        cv: Optional[int] =5,
        test_all : Optional[bool] = False,

    )->pd.DataFrame :

        n = len(self.base_forecasts.T)
        #assert that n> 1
        if not test_all :
            indexes = random.sample(range(n),cv)
        if test_all :
            indexes = np.arange(n)
            self.cv = n

        mean_score_real = 0
        mean_score_reconciled = 0

        if metrics == 'rmse':


            for index in indexes:
                mean_score_real += mean_squared_error(self.real_values[:, index], self.base_forecasts[:,index],squared=False)
                mean_score_reconciled += mean_squared_error(self.real_values[:, index], self.reconcile(method = reconcile_method,column_to_reconcile = index),squared=False)

        elif metrics=='mase':


            for index in indexes :
                mean_score_real += mean_absolute_error(self.real_values[:, index],self.base_forecasts[:,index])
                mean_score_reconciled+=mean_absolute_error(self.real_values[:, index], self.reconcile(method = reconcile_method, column_to_reconcile=index))

        elif metrics=='mse':

            for index in indexes:
                mean_score_real +=mean_squared_error(self.real_values[:,index],self.base_forecasts[:,index])
                mean_score_reconciled+=mean_squared_error(self.real_values[:,index], self.reconcile(method=reconcile_method, column_to_reconcile=index))

        else:
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'rmse','mase','mse'."
            )

        score_dataframe = pd.DataFrame(data = {metrics : [mean_score_real/self.cv,mean_score_reconciled/self.cv]})
        score_dataframe.rename(index={0: "Base forecast", 1: f"Reconciled forecast ({reconcile_method})"},inplace=True)

        return(score_dataframe)

    def plot(
        self,
        level: Optional[str] ='total',
        reconcile_method: Optional[str] = 'MinTSa', 
        columns: Optional[ArrayLike]= -1  ,
        plot_real: Optional[bool] = True,

    ) -> None :
    #asser that we hace the real values 
    #assert that plot in total +columns label ordered
        indexes_of_series = self._get_indexes_level()

        if columns==-1 :
            columns=np.arange(self.real_values.shape[1])
        plt.figure(figsize=(15,6))
        if level=='total':
            plt.plot(self.base_forecasts[0,columns],color='red')
            plt.plot(np.asarray([self.reconcile(method=reconcile_method,column_to_reconcile=i) for i in columns]).T[0],color='green')
            if plot_real :
                plt.plot(self.real_values[0,columns],color='blue')

        elif level!='total':
            for index in indexes_of_series[level] :
                plt.plot(self.base_forecasts[index,columns],color='red')
                plt.plot(np.asarray([self.reconcile(method=reconcile_method,column_to_reconcile=i) for i in columns]).T[index],color='green')
                if plot_real :
                    plt.plot(self.real_values[index,columns],color='blue')
        if plot_real :       
            plt.title(f"Real (blue), predicted (red), and reconciled with {reconcile_method} method (green) values for the {level} aggregation level")
        plt.show()


    def proba_reconcile(
        self,
        method: Optional[str] = 'MinTSh',
        alpha : Optional[float] = 0.05,
        samples_to_bootstrap : Optional[int] = -1,
        column_to_reconcile: Optional[int] = -1,
        reconcile_all: Optional[bool] =False
    ) -> None :

    #make sure the number of ssamples to bootstrap is smaller or equal tahn error matrix shape [1]

        if samples_to_bootstrap == -1 :
            samples_to_bootstrap = self.in_sample_error_matrix.shape[1]
        if column_to_reconcile ==-1 :
            column_to_reconcile = self.base_forecasts.shape[1]

        sample_base_forecasts = np.zeros(shape=(self.base_forecasts.shape[0],samples_to_bootstrap))

        for s in range(samples_to_bootstrap) :
            random_index = random.randint(0,self.in_sample_error_matrix.shape[1]-1)
            #print(sample_base_forecasts[:,s].shape)
            #print(self.base_forecasts[:,column_to_reconcile].shape)
            sample_base_forecasts[:,s] = self.base_forecasts[:,column_to_reconcile] + self.in_sample_error_matrix[:,random_index]

        sample_reconciled_forecasts = np.zeros(shape=(self.base_forecasts.shape[0],samples_to_bootstrap))

        for s in range(samples_to_bootstrap) :
            sample_reconciled_forecasts[:,s]=self.reconcile(method = method, _vector_to_proba_reconcile = sample_base_forecasts[:,s])

        test = np.quantile(sample_reconciled_forecasts,q=[0.05,0.5,0.95],axis=1)
        #print(test.shape)
        #print(self.real_values[:,column_to_reconcile])
        high=0
        low=0

        for i in range(self.base_forecasts.shape[0]):
            if (self.real_values[:,column_to_reconcile][i]<test[0,i]) :
                #print('low')
                low+=1
            if  (self.real_values[:,column_to_reconcile][i]>test[2,i]):
                #print('high')
                high+=1

        #print(f"The share of out for level {alpha} is {round((low+high)/self.base_forecasts.shape[0],3)} % with {high} forecasts outbond + and {low} outbound -")
            #print(f"The lower bound for {i} th time series is {test[0,i]} and the upper bound id {test[2,i]}" )
        score_dataframe=pd.DataFrame(data = {'lower_bound' : [test[0,i] for i in range(114)],'upper_bound' : [test[2,i] for i in range(114)]})
        print(score_dataframe)
        #score_dataframe.rename(index={0: "Base forecast", 1: f"Reconciled forecast ({reconcile_method})"},inplace=True)

        #return(test)







