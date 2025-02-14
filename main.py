import scipy.io
import pandas as pd
import numpy as np


def get_data(data_filename : str) -> np.ndarray:
    mat_data = scipy.io.loadmat(data_filename)
    X_train_df = pd.DataFrame(mat_data['X_train']).T
    X_test_df = pd.DataFrame(mat_data['X_test']).T
    y_train_df = pd.DataFrame(mat_data['y_train']).T
    y_test_df = pd.DataFrame(mat_data['y_test']).T
    return X_train_df.to_numpy(), y_train_df.to_numpy(), X_test_df.to_numpy(), y_test_df.to_numpy()


def in_interval(interval : list, indices : np.ndarray, first_inclusive : bool = True, second_inclusive : bool = False) -> bool:
    start, end = interval
    in_lower_bound = indices >= start if first_inclusive else indices > start
    in_upper_bound = indices <= end if second_inclusive else indices < end
    return in_lower_bound & in_upper_bound


def classify(X_test : np.ndarray, P_hat : np.ndarray, m : int) -> np.ndarray:
    return
    


def main():
    data_filename = "hw2_data.mat"
    X_train, y_train, X_test, y_test = get_data(data_filename)
    # m = [2,4,8,16]
    # n = list(10**i for i in range(6)) # [10, 10^2, ..., 10^6]
    
    # TESTING
    m = 4
    n = 10
    P_hat = np.empty((m,m))
    # obtain box slices
    for i in range(1, m+1):
        for j in range(1, m+1):
            S_i = [(i - 1) / m, i / m]
            S_j = [(j - 1) / m, j / m]
            
            i_idx = i - 1
            j_idx = j - 1
            
            x1_X_train = X_train[:, 0]
            x2_X_train = X_train[:, 1]
            
            indices_points_in_bin = (in_interval(S_i, x1_X_train) & in_interval(S_j, x2_X_train))
            
            bin_points = X_train[indices_points_in_bin]
            bin_points_y_train = y_train[indices_points_in_bin]
            
            num_points_in_bin = int(indices_points_in_bin.sum())
            num_bin_points_y_is_one = int(((bin_points_y_train + 1) / 2).sum())
            
            if num_points_in_bin:
                P_hat[i_idx,j_idx] = num_bin_points_y_is_one / num_points_in_bin
            else: 
                P_hat[i_idx,j_idx] = 0.5
            
            print(f"P_{i},{j} = {P_hat[i_idx,j_idx]}")
            
            
            
            

            


    

if __name__ == "__main__":
    main()
