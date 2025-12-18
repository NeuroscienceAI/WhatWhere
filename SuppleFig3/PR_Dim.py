import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def PR_Dim(X):
    # --- Participation Ratio for Effective Dimensionality ---
    temp_emb_512D = X
    print("emb shape"+ str(temp_emb_512D.shape))
    centered_data = temp_emb_512D - np.mean(temp_emb_512D, axis=0)
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]
    numerator = np.sum(eigenvalues) ** 2
    denominator = np.sum(eigenvalues ** 2)
    pr = numerator / denominator
    
    n_samples, n_neurons = temp_emb_512D.shape
    # 2. Compute Covariance Matrix and Eigenvalues
    # (We re-use these for both metrics to save time)
    cov_matrix = np.cov(centered_data, rowvar=False)
    
    # Use 'eigh' for symmetric matrices (faster/stable), returns in ascending order
    eigenvalues = np.linalg.eigh(cov_matrix)[0]
    
    # Sort descending (Largest variance first)
    eigenvalues = eigenvalues[::-1]
    
    # Filter out numerical noise (zeros/negatives)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # ==========================================================
    # Metric 1: Bias-Corrected Participation Ratio (Chun et al.)
    # ==========================================================
    trace_sigma = np.sum(eigenvalues)
    trace_sigma_squared = np.sum(eigenvalues ** 2)
    
    # Subtract sampling noise from the second moment
    bias_correction = (1.0 / n_samples) * (trace_sigma ** 2)
    corrected_second_moment = trace_sigma_squared - bias_correction
    
    # Clamp to avoid division by zero
    corrected_second_moment = max(corrected_second_moment, 1e-15)
    
    pr_corrected = (trace_sigma ** 2) / corrected_second_moment
    # ==========================================================
    # Metric 2: Dimension for Explained Variance (D_expvar)
    # ==========================================================
    # Calculate explained variance ratio
    variance_threshold=0.9
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find the number of components needed to cross the threshold
    # np.argmax returns the first index where condition is True
    # We add 1 because indices start at 0 (e.g., index 0 is the 1st dimension)
    if cumulative_variance[-1] < variance_threshold:
        d_expvar = len(eigenvalues) # Even all dims aren't enough (rare)
    else:
        d_expvar = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    return pr, pr_corrected, d_expvar
