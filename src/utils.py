import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

#==================================preprocessing of dataset functions=====================================================

def apply_scaler(df, column, df_chronic, reference_window):
    """[summary]

    Args:
        df ([type]): [description]
        column ([type]): [description]
        reference_window ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    if reference_window is None:
        reference_window = np.array([True] * df.shape[0])
        
    scaler = StandardScaler().fit(df[[column]].loc[reference_window].values)
    
    df_chronic = df_chronic.apply(lambda x: scaler.transform(x.reshape(-1,1)).ravel(), raw=True, axis=1)
    
    return df_chronic, scaler
    
   
def make_chronics(df, toshape_columns, pivot_indexcol, pivot_columncol=None):
    """[summary]

    Args:
        df ([type]): [description]
        toshape_columns ([type]): [description]
        pivot_indexcol ([type]): [description]
        pivot_columncol ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    
    assert pivot_indexcol in list(df.columns)
    if pivot_columncol is not None:
        assert pivot_columncol in list(df.columns)
    
    list_df = []

    for col in toshape_columns:
        df_pivot = df[[col, pivot_indexcol, pivot_columncol]].pivot(
                values=col, index=pivot_indexcol, columns=pivot_columncol).copy()
        
        if df_pivot.isna().sum().sum() != 0:
            df_pivot= df_pivot.interpolate(method="cubicspline", axis=0)
            
        list_df.append(df_pivot.copy())
        
    return tuple(list_df)


def make_df_calendar(df_datetime):
    """[summary]

    Args:
        df_datetime ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    ds = df_datetime.columns[0]
    df_datetime[ds] = pd.to_datetime(df_datetime[ds])
    
    df_datetime['month'] = df_datetime[ds].dt.month
    df_datetime['weekday'] = df_datetime[ds].dt.weekday
    df_datetime['is_weekend'] = (df_datetime.weekday >= 5).apply(lambda x:int(x))
    df_datetime['year'] = df_datetime[ds].dt.year
    
    return df_datetime
    
