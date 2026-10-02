"""Module for identifying principal components."""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEED = 42

def parallel_analysis(
    returns: np.ndarray, 
    n_iter: int = 1000,
    percentile: int = 95
) -> int:
    """Perform parallel analysis to determine number of principal components to analyze 
    in the data:
    1. Generate eigenvalues from correlation structure in the data
    2. Simulate `n_iter` random variations of the data
    3. Keep the eigenvalues that beat the 95th percentile of their corresponding random eigenvalue    

    Args:
        returns (np.ndarray): Scaled monthly factor returns across a period.
        n_iter (int): The number of iterations to run.
        percentile (int): The percentile to beat. 

    Returns:
        int: The number of components to use for PCA analysis.
    """
    np.random.seed(SEED)
    returns = returns.to_numpy()
    T, K = returns.shape

    # Get eigenvalues from correlation matrix
    real_eigenvalues = np.linalg.eigvalsh(np.corrcoef(returns.T))[::-1]
    
    null_eigenvalues = np.zeros((n_iter, K))
    for i in range(n_iter):
        # Randomly generate a permutation of factor returns
        permuted = np.apply_along_axis(np.random.permutation, 0, returns)
        null_eigenvalues[i] = np.linalg.eigvalsh(np.corrcoef(permuted.T))[::-1]
    
    # Keep the real eigenvalue beat the 95th percentile of what the 
    # k-th eigenvalue looks like under no correlation
    thresholds = np.percentile(null_eigenvalues, percentile, axis=0)
    return int(np.sum(real_eigenvalues > thresholds))

def view_principal_components(returns: pd.DataFrame) -> pd.DataFrame:
    """Perform PCA to see how factors contribute principal component loadings.
    Conceptually, PCA determines which underlying "axes" drive the most variation
    in the data. By performing PCA on monthly factor returns, I am hoping to see 
    how factors group together.

    Args:
        returns: Chronologically sorted monthly factor returns across a period.

    Returns:
        pd.DataFrame: A DataFrame holding principal component loadings.
    """
    factor_names = returns.columns

    # Scale data before parallel analysis + PCA
    scaler = StandardScaler()
    returns = scaler.fit_transform(returns)

    n_components = parallel_analysis(returns, n_iter=1000, percentile=95)
    pca = PCA(n_components=n_components)
    pca.fit(returns)

    # Extract loadings
    loadings = pd.DataFrame(
        pca.components_.T,          
        index=factor_names,         
        columns=[f'PC{i+1}' for i in range(n_components)],
    )

    order = loadings['PC1'].abs().sort_values(ascending=False).index
    return loadings.loc[order]