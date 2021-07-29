from typing import Optional, Union
import numpy as np 
import pdb 
from pdb import set_trace
from numpy.lib import diag
import pandas as pd
from typing import Optional, Union, Iterable, Tuple, List
from numpy.typing import ArrayLike
from sklearn.metrics import mean_squared_error



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
        reconciled_forecasts : ArrayLike = None
    ) -> None:
        self.data = data
        self.summing_mat = summing_mat
        self.in_sample_error_matrix = in_sample_error_matrix
        self.base_forecasts = base_forecasts
        self.columns_labels_ordered = columns_labels_ordered
        self.reconciled_forecasts = reconciled_forecasts
        self.real_values = real_values

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


    def get_number_bottom_series(
        self,
    ) -> int :
        #print(self.summing_mat)
        n = (self.summing_mat).shape[0]
        i=0
        while self.summing_mat[i].sum() != 1 :
            i+=1
        return (n-i)

    def reconcile(     
        self,
        method: str
    ) -> ArrayLike :
        
        if method=='OLS' :  #"OLS" "Variance_scaling" "Structural_Scaling" " Mint_Sample" "MinT_Shrink" "Top_down"

            combination_matrix = np.linalg.inv(np.transpose(self.summing_mat)@self.summing_mat)@np.transpose(self.summing_mat)                   
        
        elif method=='BU' :  #"OLS" "Variance_scaling" "Structural_Scaling" " Mint_Sample" "MinT_Shrink" "Top_down"

            combination_matrix = np.concatenate((np.zeros(shape=(self.get_number_bottom_series(),self.summing_mat.shape[0]-self.get_number_bottom_series())),np.identity(self.get_number_bottom_series())), axis=1)

        elif method =='SS' : #Structural Scaling

            W=np.diag(self.summing_mat@np.ones(self.get_number_bottom_series()))
            combination_matrix = np.linalg.inv((self.summing_mat.T)@np.linalg.inv(W)@self.summing_mat)@(self.summing_mat.T)@np.linalg.inv(W)

        elif method in ['VS','MinTSa','MinTSh']:

            number_error_vectors = self.in_sample_error_matrix.shape[1]
            #print(number_error_vectors)
            W1=np.zeros((self.summing_mat.shape[0],self.summing_mat.shape[0]))
            #return(self.in_sample_error_matrix[:,1][:,None])@(self.in_sample_error_matrix[:,1][:,None].T)
            for i in range(number_error_vectors):
                W1+=(self.in_sample_error_matrix[:,i][:,None])@(self.in_sample_error_matrix[:,i][:,None].T)
            
            W1=W1/number_error_vectors

            if method == 'VS' :                
                W=np.diag(np.diag(W1))
            elif method == 'MinTSa' :
                W=W1
            elif method == 'MinTSh' :
                W= W1 #TODO MINTSHRINK

            combination_matrix = np.linalg.inv((self.summing_mat.T)@np.linalg.inv(W)@self.summing_mat)@(self.summing_mat.T)@np.linalg.inv(W)

        else :
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'OLS','BU', SS', 'VS', 'MinTSa' and 'MinTSh'."
            )

        self.reconciled_forecasts = self.summing_mat@combination_matrix@self.base_forecasts

        return(self.reconciled_forecasts)




    def score(
        self,
        metrics:str
    ) -> pd.DataFrame :

        if metrics=='rmse' :
            score_dataframe=pd.DataFrame(data = {'rmse' : [mean_squared_error(self.real_values,self.base_forecasts),mean_squared_error(self.real_values,self.reconciled_forecasts)]})

        return(score_dataframe)


    
    
    
    
