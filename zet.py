'''
Missing values imputer for 2D arrays {n_samples,n_features} or pandas DataFrames.

Based on ZET algorithm (Zagoruiko, 1999)
'''

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def zet_fill(df,scale='standard',cmshape=5,alpha=3):
    '''
    Input:
    df - 2D array or pandas.DataFrame
    scale - {'standard','minmax','none'} method for scaling features
    cmshape - size of compact submatrix {cmshape,cmshape}
    alpha - contribution power of rows (columns) competence
    
    Returns:
    2D array of the same size with impted missing values   
    '''
    
    def compet(X,Y):
        '''
        rows competence calculation
        X,Y - np.arrays
        '''
        mask = pd.notnull(X) & pd.notnull(Y)
        t = len(X[mask])
        if t > 0:
            r = 1/euclidean(X[mask],Y[mask])
            return r*t
        else:
            return 0

    def compet_corr(X,Y):
        '''
        columns competence calculation
        X,Y - np.arrays
        '''
        mask = pd.notnull(X) & pd.notnull(Y)
        t = len(X[mask])
        if t > 1:
            r = abs(np.corrcoef(X[mask],Y[mask])[0,1])
            return r*t
        elif t == 1:
            return 1.0e-15
        else:
            return 0

    def regress(X,Y,i):
        '''
        Linear regression to compute Y[i] value
        '''
        mask = pd.notnull(X) & pd.notnull(Y)
        if len(X[mask]) > 0:
            lr = LinearRegression()
            lr.fit(X[mask].values.reshape(-1,1),Y[mask])
            return lr.predict(X[i])
        else:
            return 0
        
        
    df = np.array(df)
    nan_args = np.argwhere(pd.isnull(df))
    
    if scale == 'minmax':
        mmsc = MinMaxScaler()
        dfs = mmsc.fit_transform(pd.DataFrame(df).fillna(pd.DataFrame(df).mean()))
        dfs[pd.isnull(df)] = np.nan
        dfs_copy = dfs.copy()
    elif scale == 'standard':
        ssc = StandardScaler()
        dfs = ssc.fit_transform(pd.DataFrame(df).fillna(pd.DataFrame(df).mean()))
        dfs[pd.isnull(df)] = np.nan
        dfs_copy = dfs.copy()
    else:
        dfs = df.copy()
        dfs_copy = df.copy()
    

    N = df.shape[0]
    M = df.shape[1]

    for x,y in nan_args:
        tmp = pd.DataFrame(dfs)

        A = tmp.loc[x]
        comp_rows = pd.Series(index = range(N))
        for i in range(N):
            B = tmp.loc[i]
            if i != x and pd.notnull(B[y]):
                comp_rows[i] = compet(A,B)
        am = min(len(comp_rows[comp_rows>0]),cmshape-1)
        comp_indx = np.append(comp_rows.drop(x).sort_values(ascending=False).index[:am],x)
        tmp = tmp.loc[comp_indx].transpose()



        C = tmp.loc[y]
        comp_cols = pd.Series(index = range(M))
        for j in range(M):
            D = tmp.loc[j]
            if j != y and pd.notnull(D[x]):
                comp_cols[j] = compet_corr(C,D)
        am = min(len(comp_cols[comp_cols>0]),cmshape-1)
        comp_indy = np.append(comp_cols.drop(y).sort_values(ascending=False).index[:am],y)
        tmp = tmp.loc[comp_indy].transpose()

        if tmp.shape[0] > 1 and tmp.shape[1] > 1:

            sum_rows = 0
            sum_compx = 0
            Y = tmp.loc[x]
            for ci in comp_indx:
                if ci != x:
                    X = tmp.loc[ci]
                    sum_rows += regress(X,Y,y)*comp_rows[ci]**alpha
                    sum_compx += comp_rows[ci]**alpha

            pred_r = sum_rows/sum_compx

            tmp = tmp.transpose()
            sum_cols = 0
            sum_compy = 0
            Y = tmp.loc[y]
            for cj in comp_indy:
                if cj != y:
                    X = tmp.loc[cj]
                    sum_cols += regress(X,Y,x)*comp_cols[cj]**alpha
                    sum_compy += comp_cols[cj]**alpha

            pred_c = sum_cols/sum_compy

            pred = 0.5*(pred_r + pred_c)

            tmp = tmp.transpose()

        else:
            pred = np.nan
            print('Warning: unable to make prediction!')

        dfs_copy[x,y] = pred
        
    if scale == 'minmax':
        return mmsc.inverse_transform(dfs_copy)
    elif scale == 'standard':
        return ssc.inverse_transform(dfs_copy)
    else:
        return dfs_copy
