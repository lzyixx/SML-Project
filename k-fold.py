import pandas as pd
import numpy as np

def k_fold_with_direction(train_df,k,random_seed=42):
    np.random.seed(random_seed)
    
    indices = np.arange(len(train_df))      
    np.random.shuffle(indices)              

    fold_size = len(train_df) // k

    for fold in range(k):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold != k - 1 else len(train_df)

        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        yield fold_train,fold_val,fold + 1

def seprate_train_test(test_region):
    df = pd.read_csv("dataset.csv")

    test_df = df[df["Direction_" + test_region] == 1]
    train_df = df[df["Direction_" + test_region] == 0]

    return train_df,test_df

# how to use
train_X,test_X = seprate_train_test("West")
for train_df, val_df, fold in k_fold_with_direction(train_X, 5):
    print(len(train_df))
    print(len(val_df))


