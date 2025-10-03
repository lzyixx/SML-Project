import numpy as np
import pandas as pd


def temporal_random_split_by_horizon(train_size=0.8, vali_size=0.2,test_size=0.2, random_seed=42):
    """
    using temporal dataset cross-validation method.

    divide dataset with 50% : 50%

    sample randomly from first half part for train, sample randomly from second half part for test

    casue there are 10 company stock data in our dataset, there are 10 lines in each date.
    so 10 lines represent 1 instances.

    Input parameters:
        train_size: float, default=0.8
            Proportion of days to sample from the first half as the training set.For example, 0.8 means 80% of the first-half days are selected.
        
        test_size: float, default=0.2
            Proportion of days to sample from the second half as the test set.For example, 0.2 means 20% of the second-half days are selected.
        
        random_seed: int, default=42
            Random seed for reproducibility.

    return:
        train_samples : pd.DataFrame
            Training samples (all features, without the helper column Date_id).

        test_samples : pd.DataFrame
            Testing samples (all features, without the helper column Date_id).
    
    """
    if not (0 < train_size <= 1):
        raise ValueError(f"train_size must be between 0 and 1, got {train_size}")
    if not (0 < test_size <= 1):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if not (0 < vali_size <= 1):
        raise ValueError(f"vali_size must be between 0 and 1, got {test_size}")
    if train_size + vali_size > 1:
        raise ValueError(
            f"train_size + vali_size cannot exceed 1, got {train_size + vali_size}"
        )
    if not isinstance(random_seed, int):
        raise TypeError(f"random_seed must be an integer, got {type(random_seed)}")


    # set random seed for randomly samples from dataset
    np.random.seed(random_seed)

    df = pd.read_csv("processed_data.csv")

    # add columns for date_id
    data = df.copy()
    data["Date_id"] = data["Year"].astype(str) + "_" + data["DayofYear"].astype(str)

    unique_days = data["Date_id"].unique()
    n_days = len(unique_days)


    
    mid = n_days //2
    three_quater = mid + mid//2
    
    first_half_days = unique_days[:mid]
    second_half_days_test = unique_days[three_quater:]
    second_half_days_vali = unique_days[mid:three_quater]
    

    train_num_samples = int(mid * train_size)
    test_num_samples = int(mid * test_size)
    vali_num_samples = int(mid * vali_size)
    train_samples_id = np.random.choice(first_half_days, size=train_num_samples, replace=False)
    test_samples_id = np.random.choice(second_half_days_test, size=test_num_samples, replace=False)
    vali_samples_id = np.random.choice(second_half_days_vali, size=vali_num_samples, replace=False)

    train_samples = data[data["Date_id"].isin(train_samples_id)].drop(columns=["Date_id"])
    test_samples = data[data["Date_id"].isin(test_samples_id)].drop(columns=["Date_id"])
    vali_samples = data[data["Date_id"].isin(vali_samples_id)].drop(columns=["Date_id"])
    
    
    return train_samples,test_samples,vali_samples
