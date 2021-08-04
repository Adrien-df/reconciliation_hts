from typing import Optional, Union
import numpy as np 
import pdb 
from pdb import set_trace
from numpy.lib import diag
import pandas as pd
from typing import Optional, Union, Iterable, Tuple, List
from numpy.typing import ArrayLike
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import random
import utils
import matplotlib.pyplot as plt
from statsmodels.stats.moment_helpers import cov2corr




class To_Reconcile:
    """[Test of docstring]

    [This is the description of the class that is realised thanks to the docstring]
    """
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        summing_mat: Optional[np.ndarray] = None,
        in_sample_error_matrix: Optional[ArrayLike] = None,
        base_forecasts: Optional[ArrayLike] = None,
        columns_labels_ordered: Optional[list[str]] = None,
        real_values: Optional[ArrayLike] = None,
        #reconciled_forecasts : ArrayLike = None
        lambd = None
    ) -> None:
        self.data = data
        self.summing_mat = summing_mat
        self.in_sample_error_matrix = in_sample_error_matrix
        self.base_forecasts = base_forecasts
        self.columns_labels_ordered = columns_labels_ordered
        #self.reconciled_forecasts = reconciled_forecasts
        self.real_values = real_values
        self.lambd = lambd

    def _check_parameters(self)->None :
        """[summary]

        [extended_summary]

        Raises
        ------
        ValueError
            [description]
        ValueError
            [description]
        ValueError
            [description]
        """
        if not isinstance(self.summing_mat,np.ndarray):
            raise ValueError(
                "Invalid type for summing matrix. Expected numpy array type "
            )

        if len(self.summing_mat)<3 :
            raise ValueError(
                "Invalid summing_mat shape. Msut be of shape >= 3."
            )
        if len(np.unique(self.summing_mat)) != 2 :
            raise ValueError(
                "Invalid summing_mat argument. Must contain zero and ones only."
            )

        
    def compute_summing_mat(
        self
    ) -> np.ndarray:

        L=len(self.columns_labels_ordered) #assert that the lenght is sufficient
        n = self.data.shape[0]
        m = self.data[self.columns_labels_ordered].isna().any(axis=1).sum()
        self.summing_mat = np.zeros([n,n-m])
        self.summing_mat[0]=np.ones(n-m)
        self.summing_mat[m:]=np.identity(n-m)
    
    
    
        for level in range(1,L):
        
           for i in range(1,n-m):
            
                    if pd.isna(self.data.iloc[i][self.columns_labels_ordered[level]]) and not pd.isna(self.data.iloc[i][self.columns_labels_ordered[level-1]]) :
                        list_of_values = [self.data.iloc[i][self.columns_labels_ordered[k]] for k in range(level)]
                    
                    
                        for j in range(m,n):
                            if [self.data.iloc[j][self.columns_labels_ordered[k]] for k in range(level)] == list_of_values:
                                self.summing_mat[i][j-m]=1
        return(self.summing_mat)

    def _get_indexes_level(
        self
    )->dict :
        dictionnary={'total' : 0}
        L=len(self.columns_labels_ordered) #assert that the lenght is sufficient
        n = self.data.shape[0]

        for level in range(1,L):
            indexes_of_level =[]
            for i in range(1,n):
                
                if pd.isna(self.data.iloc[i][self.columns_labels_ordered[level]]) and not pd.isna(self.data.iloc[i][self.columns_labels_ordered[level-1]]) :
                    indexes_of_level.append(i)
        
            dictionnary[self.columns_labels_ordered[level-1]]=indexes_of_level  
        indexes_of_level = []
        for i in range(1,n):
            if not pd.isna(self.data.iloc[i][self.columns_labels_ordered[-1]]):
                indexes_of_level.append(i)
        dictionnary[self.columns_labels_ordered[-1]]=indexes_of_level           
                
            
        return(dictionnary)                     
                        


    def get_number_bottom_series(
        self,
    ) -> int :
        n = (self.summing_mat).shape[0]
        i=0
        while self.summing_mat[i].sum() != 1 :
            i+=1
        return (n-i)

    def _compute_lambda(
        self
    )->float :
        res=np.matrix(self.in_sample_error_matrix).T
        number_error_vectors = res.shape[0]
        covm = utils.cross_product(res)/number_error_vectors
        corm = cov2corr(covm)
        xs = utils.scale(res,np.sqrt(np.diag(covm)))
        v= (1/(number_error_vectors * (number_error_vectors - 1))) * (utils.cross_product(np.square(np.matrix(xs))) - 1/number_error_vectors * (np.square(utils.cross_product(np.matrix(xs))) ))
        np.fill_diagonal(v,0)
        corapn = cov2corr(np.diag(np.diag(covm)))
        d = np.square((corm - corapn))
        lambd = v.sum()/d.sum()
        lambd= max(min(lambd,1), 0)
        self.lambd = lambd
        return(lambd)

    def reconcile(     
        self,
        method: Optional[str] = 'MintSa',
        column_to_reconcile: Optional[int] = -1,
        reconcile_all: Optional[bool] =False,
        _vector_to_proba_reconcile : Optional[np.ndarray] = None,
        show_lambda: Optional[bool] = False
    ) -> ArrayLike :
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

        if self.summing_mat is None :
            self.summing_mat = self.compute_summing_mat()
        #Faire tous les checks, notamment les checks de combinaison 
        
        if method=='OLS' :  #"OLS" "Variance_scaling" "Structural_Scaling" " Mint_Sample" "MinT_Shrink" "Top_down"

            combination_matrix = np.linalg.inv(np.transpose(self.summing_mat)@self.summing_mat)@np.transpose(self.summing_mat)                   
        
        elif method=='BU' :  #"OLS" "Variance_scaling" "Structural_Scaling" " Mint_Sample" "MinT_Shrink" "Top_down"

            combination_matrix = np.concatenate((np.zeros(shape=(self.get_number_bottom_series(),self.summing_mat.shape[0]-self.get_number_bottom_series())),np.identity(self.get_number_bottom_series())), axis=1)

        elif method =='SS' : #Structural Scaling

            W=np.diag(self.summing_mat@np.ones(self.get_number_bottom_series()))
            combination_matrix = np.linalg.inv((self.summing_mat.T)@np.linalg.inv(W)@self.summing_mat)@(self.summing_mat.T)@np.linalg.inv(W)

        elif method in ['VS','MinTSa','MinTSh']:

            number_error_vectors = self.in_sample_error_matrix.shape[1]
            W1=np.zeros((self.summing_mat.shape[0],self.summing_mat.shape[0]))
            for i in range(number_error_vectors):
                W1+=(self.in_sample_error_matrix[:,i][:,None])@(self.in_sample_error_matrix[:,i][:,None].T)
            
            W1=W1/number_error_vectors
            #print(W1)

            if method == 'VS' :                
                W=np.diag(np.diag(W1))
            elif method == 'MinTSa' :
                W=W1
            elif method == 'MinTSh' :
                if self.lambd ==None :
                    
                    lambd = self._compute_lambda()
                    if show_lambda :
                        print("lambda parameter for MinTSh reconciliation method is equal to : ",lambd)
                else : 
                    lambd = self.lambd
                
                W=lambd*np.diag(np.diag(W1)) + (1-lambd)*W1

            combination_matrix = np.linalg.inv((self.summing_mat.T)@np.linalg.inv(W)@self.summing_mat)@(self.summing_mat.T)@np.linalg.inv(W)

        else :
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'OLS','BU', SS', 'VS', 'MinTSa' and 'MinTSh'."
            )

        #self.reconciled_forecasts = self.summing_mat@combination_matrix@self.base_forecasts
        if _vector_to_proba_reconcile is not None :
            return(self.summing_mat@combination_matrix@_vector_to_proba_reconcile)
        elif reconcile_all :
            return( [self.summing_mat@combination_matrix@self.base_forecasts[:,i] for i in len(self.base_forecasts)])
        elif column_to_reconcile==-1 :
            return(self.summing_mat@combination_matrix@self.base_forecasts)
        else :
            return(self.summing_mat@combination_matrix@self.base_forecasts[:,column_to_reconcile])




    def score(
        self,
        metrics: Optional[str] ='rmse',
        reconcile_method: Optional[str] = 'MinTSa',
        
    ) -> pd.DataFrame :

    #check that there is only one vector in base forecasts

        if metrics=='rmse' :
            score_dataframe=pd.DataFrame(data = {'rmse' : [mean_squared_error(self.real_values,self.base_forecasts,squared=False),
            mean_squared_error(self.real_values,self.reconcile(method=reconcile_method),squared=False)]})

        elif metrics=='mse' :
            score_dataframe=pd.DataFrame(data = {'rmse' : [mean_squared_error(self.real_values,self.base_forecasts,squared=False),
            mean_squared_error(self.real_values,self.reconcile(method=reconcile_method),squared=False)]})

        elif metrics=='mase' :
            score_dataframe=pd.DataFrame(data = {'mase' : [mean_absolute_error(self.real_values,self.base_forecasts),
            mean_absolute_error(self.real_values,self.reconcile(method=reconcile_method))]})

        score_dataframe.rename(index={0: "Base forecast", 1: f"Reconciled forecast ({reconcile_method})"},inplace=True)
        
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

        mean_score_real =0
        mean_score_reconciled = 0

        if metrics=='rmse' :

            
            for index in indexes :
                mean_score_real +=mean_squared_error(self.real_values[:,index],self.base_forecasts[:,index],squared=False)
                mean_score_reconciled+=mean_squared_error(self.real_values[:,index],self.reconcile(method=reconcile_method,column_to_reconcile=index),squared=False)

        elif metrics=='mase' :

            
            for index in indexes :
                mean_score_real +=mean_absolute_error(self.real_values[:,index],self.base_forecasts[:,index])
                mean_score_reconciled+=mean_absolute_error(self.real_values[:,index],self.reconcile(method=reconcile_method,column_to_reconcile=index))

        elif metrics=='mse' :

            for index in indexes :
                mean_score_real +=mean_squared_error(self.real_values[:,index],self.base_forecasts[:,index])
                mean_score_reconciled+=mean_squared_error(self.real_values[:,index],self.reconcile(method=reconcile_method,column_to_reconcile=index))
    
        else :
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'rmse','mase','mse'."
            )
    
        score_dataframe=pd.DataFrame(data = {metrics : [mean_score_real/self.cv,mean_score_reconciled/self.cv]})
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
        print(test.shape)
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

        print(f"The share of out for level {alpha} is {round((low+high)/self.base_forecasts.shape[0],3)} % with {high} forecasts outbond + and {low} outbound -")
            #print( test[0,i],self.real_values[:,column_to_reconcile],test[2,i])
        

        #return(test)


    
        
        

            

