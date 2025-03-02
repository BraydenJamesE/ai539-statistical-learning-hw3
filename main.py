import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_data(data_filename : str) -> np.ndarray: # getting the data from the .mat file, converting it to a DataFrame and then return the np.ndarray
    mat_data = scipy.io.loadmat(data_filename)
    X_train_df = pd.DataFrame(mat_data['X_train']).T
    X_test_df = pd.DataFrame(mat_data['X_test']).T
    y_train_df = pd.DataFrame(mat_data['y_train']).T
    y_test_df = pd.DataFrame(mat_data['y_test']).T
    bayes_risk_df = pd.DataFrame(mat_data['Rmin'])
    return X_train_df.to_numpy(), y_train_df.to_numpy(), X_test_df.to_numpy(), y_test_df.to_numpy(), bayes_risk_df.to_numpy()


def is_in_interval(interval_i : list, interval_j : list, indices : np.ndarray, first_inclusive : bool = True, second_inclusive : bool = False) -> bool: # This function checks if a specific data poitn or a list of datapoints are in the iteration. If it's one datapoint, the function returns True or False. If it's a list of datapoints, the function will return a list of True or False answers for the datapoint at that index. 
    start_i, end_i = interval_i
    start_j, end_j = interval_j

    if isinstance(indices, list) or indices.ndim == 1: # checking if the input is a true list of a tuple
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


def get_bin_indices(data_point : np.ndarray, m : int) -> tuple: # after vectorizing, I did not end up using this function
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
            
            bin_points_y_train = y_train[indices_points_in_bin]
            
            num_points_in_bin = int(indices_points_in_bin.sum())
            num_bin_points_y_is_one = int(((bin_points_y_train + 1) / 2).sum())
            
            if num_points_in_bin:
                P_hat[i_idx,j_idx] = num_bin_points_y_is_one / num_points_in_bin
            else: 
                P_hat[i_idx,j_idx] = 0.5

    return P_hat


def get_bayes_classifier(predictions : np.ndarray) -> np.ndarray:
    random_choice = np.random.choice([-1, 1], size=predictions.shape)
    classification = np.where(predictions > 0.5, 1, np.where(predictions < 0.5, -1, random_choice)) # computing the classification
    return classification


def get_n_random_data_points(data : np.ndarray, n : int) -> np.ndarray:
    random_indices = np.random.choice(data.shape[0], size=n, replace=False)
    return data[random_indices]


def get_expected_risk(classifications : np.ndarray, true_targets : np.ndarray) -> float: # computing expected risk (incorrect / total)
    total_values = classifications.shape[0]
    total_incorrect_predictions = (classifications != true_targets).sum()
    return total_incorrect_predictions / total_values


def plot_scatter_excess_risk(excess_risk_results_all : np.ndarray, n_array : np.ndarray, m_array : np.ndarray) -> None:  # all risk is a 3D array with the size of (len(m_array), len(n_array), monte_carlo amount)

    for m_idx, m in enumerate(m_array): 
        plt.figure(figsize=(8, 6))
        all_excess_risk = np.array(excess_risk_results_all[m_idx]) # shape is (len(n_array), monte_carlo amount)
        for n_idx, n in enumerate(n_array):
            plt.loglog([n] * all_excess_risk.shape[1], all_excess_risk[n_idx], 'o', markersize=10, alpha=0.8, color=f"C{m_idx}", label=None)

        sorted_risk = np.sort(all_excess_risk, axis=1)
        lower_bound = sorted_risk[:, 4] # fifth smallest
        upper_bound = sorted_risk[:, -5] # fifth largest value

        plt.loglog(n_array, lower_bound, linestyle="dashed", label=f"m = {m} Lower", color="darkred")
        plt.loglog(n_array, upper_bound, linestyle="dashed", label=f"m = {m} Upper", color="darkred")

        plt.xlabel("Number of Training Samples (n)")
        plt.ylabel("Excess Risk")
        plt.title(f"Excess Risk Scatter Plot for m = {m}")
        plt.legend()
        plt.grid()
        plt.show()


def plot_average_excess_risk(risk_results : np.ndarray, n_array : np.ndarray, m_array : np.ndarray) -> None: 
    plt.figure(figsize=(8, 6))

    for i, m in enumerate(m_array): 
        plt.loglog(n_array, risk_results[i], label=f"m = {m}", marker='o')
    
    plt.xlabel("Number of Training Samples (n)")
    plt.ylabel("Average Excess Risk")
    plt.title("Average Excess Risk vs. Training Sample Size")
    plt.legend()
    plt.grid()
    plt.show()


def get_predictions_in_chunks(X_subset, P_hat, m, chunk_size=100000):
    predictions = []

    for start in range(0, X_subset.shape[0], chunk_size): # iterating through the number of rows in X_subset with a stepsize of chunck_size
        end = min(start + chunk_size, X_subset.shape[0]) # creating the end index 
        data_chunk = X_subset[start:end] 
        # Compute bin indices for the chunk
        bin_indices = np.floor(data_chunk * m).astype(int) # computing the index by multiplying the data points by m and casting that to an integer
        bin_indices = np.minimum(bin_indices, m - 1) # ensuring that the indexes are m - 1
        # Retrieve predictions for this chunk
        chunk_predictions = P_hat[bin_indices[:, 0], bin_indices[:, 1]]  
        predictions.append(chunk_predictions)

    return np.concatenate(predictions)


def main():
    # obtaining database information
    data_filename = "hw2_data.mat"
    X_train, y_train, X_test, y_test, bayes_risk_array = get_data(data_filename)
    bayes_risk = bayes_risk_array.item()

    m_array = [2,4,8,16] # Number of bins
    n_array = list(10**i for i in range(1, 7)) # Number of data points = [10, 10^2, ..., 10^6]

    test_subset_size = X_test.shape[0] // 5 # getting 1/10 of the test data

    num_monte_carlo_runs = 100 

    avg_excess_risk_results = np.empty((len(m_array), len(n_array)))
    excess_risk_results_all = np.empty((len(m_array), len(n_array), num_monte_carlo_runs))

    for m_idx, m in enumerate(m_array):
        for n_idx, n in enumerate(n_array):

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
                random_indices = np.random.choice(X_test.shape[0], size=test_subset_size, replace=False) # obtaining a random subset of the test data 

                X_subset = X_test[random_indices]  
                predictions = get_predictions_in_chunks(X_subset, P_hat, m)
                predictions = predictions.reshape(-1, 1)
                classifications = get_bayes_classifier(predictions)
                expected_risk = get_expected_risk(classifications, y_test[random_indices]) 
                            
                excess_risk_results_all[m_idx, n_idx, i] = abs(bayes_risk - expected_risk)

            avg_excess_risk_results[m_idx, n_idx] = np.mean(excess_risk_results_all[m_idx, n_idx, :])
            print(f"Average Excess Risk for m and n: {np.average(excess_risk_results_all[m_idx, n_idx, :]):.3f}") 
            print("-------------------")


    plot_average_excess_risk(avg_excess_risk_results, n_array, m_array)
    plot_scatter_excess_risk(excess_risk_results_all, n_array, m_array)



if __name__ == "__main__":
    main()
