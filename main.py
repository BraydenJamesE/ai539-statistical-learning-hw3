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


def is_in_interval(interval_i : list, interval_j : list, indices : np.ndarray, first_inclusive : bool = True, second_inclusive : bool = False) -> bool: 
    
    start_i, end_i = interval_i
    start_j, end_j = interval_j

    if isinstance(indices, list) or indices.ndim == 1:
        x_i_idx, x_j_idx = indices
    else: 
        x_i_idx = indices[:, 0]
        x_j_idx = indices[:, 1]

    in_lower_bound_i = x_i_idx >= start_i if first_inclusive else x_i_idx > start_i
    in_upper_bound_i = x_i_idx <= end_i if second_inclusive else x_i_idx < end_i
    
    in_lower_bound_j = x_j_idx >= start_j if first_inclusive else x_j_idx > start_j
    in_upper_bound_j = x_j_idx <= end_j if second_inclusive else x_j_idx < end_j

    is_point_in_interval = in_lower_bound_i & in_upper_bound_i & in_lower_bound_j & in_upper_bound_j

    return is_point_in_interval


def get_bin_indices(data_point : np.ndarray, m : int) -> tuple:
    for i in range(1, m+1):
        for j in range(1, m+1):
            S_i = [(i - 1) / m, i / m]
            S_j = [(j - 1) / m, j / m]
            
            if is_in_interval(S_i, S_j, data_point):
                return i-1, j-1
    
    raise ValueError(f"Data point {data_point} does not belong to bin.")


def get_probability_estimation(X_train : np.ndarray, y_train : np.ndarray, m : int) -> np.ndarray:
    
    P_hat = np.empty((m,m))

    for i in range(1, m+1):
        for j in range(1, m+1):
            S_i = [(i - 1) / m, i / m]
            S_j = [(j - 1) / m, j / m]
            
            i_idx = i - 1
            j_idx = j - 1
            
            indices_points_in_bin = is_in_interval(S_i, S_j, X_train)
            
            bin_points = X_train[indices_points_in_bin]
            bin_points_y_train = y_train[indices_points_in_bin]
            
            num_points_in_bin = int(indices_points_in_bin.sum())
            num_bin_points_y_is_one = int(((bin_points_y_train + 1) / 2).sum())
            
            if num_points_in_bin:
                P_hat[i_idx,j_idx] = num_bin_points_y_is_one / num_points_in_bin
            else: 
                P_hat[i_idx,j_idx] = 0.5

    return P_hat


def get_bayes_classifier(predictions : np.ndarray) -> np.ndarray:
    classification = np.where(predictions > 0.5, 1, np.where(predictions < 0.5, -1, np.random.choice([-1, 1], size=predictions.shape))) # computing the classification
    return classification


def get_n_random_data_points(data : np.ndarray, n : int) -> np.ndarray:
    random_indices = np.random.choice(data.shape[0], size=n, replace=False)
    return data[random_indices]


def get_expected_risk(classifications : np.ndarray, true_targets : np.ndarray) -> float:
    total_values = classifications.shape[0]
    correct_predictions = classifications == true_targets
    total_correct_predictions = correct_predictions.sum()
    return total_correct_predictions / total_values


def main():
    data_filename = "hw2_data.mat"
    X_train, y_train, X_test, y_test = get_data(data_filename)
    m_array = [2,4,8,16]
    n_array = list(10**i for i in range(6)) # [10, 10^2, ..., 10^6]

    for m in m_array:
        for n in n_array:
            num_monte_carlo_runs = 100
            all_expected_risks = []

            print(f"Expected Risk:")
            print(f"(m = {m}, n = {n})") 

            for i in range(num_monte_carlo_runs):
                # obtaining sampled data
                random_indices = np.random.choice(X_train.shape[0], size=n, replace=False)
                sampled_X_train = X_train[random_indices]
                sampled_y_train = y_train[random_indices]

                # obtaining model
                P_hat = get_probability_estimation(sampled_X_train, sampled_y_train, m) # getting probability of y = 1 per bin based on data

                predictions = [] # initializing my predictions for the test data
                for x1,x2 in X_test: # looping through each point in the sampled train data and getting thier bin index

                    data_point = [x1, x2]
                    i_idx, j_idx = get_bin_indices(data_point, m)

                    predictions.append(P_hat[i_idx][j_idx]) # appending their p_hat values for that bin 

                predictions = np.array(predictions).reshape(-1,1)
                classifications = get_bayes_classifier(predictions)
                expected_risk = get_expected_risk(classifications, y_test)
                
                print(f"{i}: {expected_risk}")
                
                all_expected_risks.append(expected_risk)
                
            print(f"Average Risk for m and n: {np.average(all_expected_risks)}")



if __name__ == "__main__":
    main()
