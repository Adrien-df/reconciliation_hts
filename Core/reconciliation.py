from typing import Optional, Union
import numpy as np 
import pdb 
from pdb import set_trace
import pandas as pd
from typing import Optional, Union, Iterable, Tuple, List
from numpy.typing import ArrayLike



def reconcile(summing_mat,in_sample_error_matrix,base_forecasts,method) :
    """[summary]

    [extended_summary]

    Parameters
    ----------
    summing_mat : [type]
        [description]
    in_sample_error_matrix : [type]
        [description]
    base_forecasts : [type]
        [description]
    method : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # "OLS" "Variance_scaling" "Structurl_Scaling" " Mint_Sample" "MinT_Shrink" "Top_down"
    n = summing_mat.shape[1]
    # set_trace()
    print("The number of series is ", n)
    return (n)


if __name__ == "__main__":
    #reconcile(np.matrix([[1, 2], [3, 4]]))
    print('ok il ny a pas  lézardo')

    def test(a,b,c):
        """[summary]

        [extended_summary]

        Parameters
        ----------
        a : [type]
            [description]
        b : [type]
            [description]
        c : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        return (a+b+c)


class To_Reconcile:
    def __init__(
        self,
        summing_mat: ArrayLike,
        in_sample_error_matrix: Optional[ArrayLike],
        base_forecasts: ArrayLike
    ) -> None:
        self.summing_mat = summing_mat,
        self.in_sample_error_matrix = in_sample_error_matrix,
        self.base_forecasts = base_forecasts

    def ma_fonction_lezardo(a:[int], b:[ArrayLike])-> double :
        """[summary]

        [extended_summary]

        Parameters
        ----------
        a : [type]
            [description]
        b : [type]
            [description]

        Returns
        -------
        double
            [description]
        """
        return (a+b)