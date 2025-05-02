# based on https://github.com/dawenl/vae_cf

import os
import sys

import numpy as np
from scipy import sparse
import pandas as pd

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process dataset for VAE_CF")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed data')
    parser.add_argument('--threshold', type=float, required=True, help='Rating threshold to filter data')
    parser.add_argument('--min_items_per_user', type=int, default=5, help='Minimum number of items per user')
    parser.add_argument('--min_users_per_item', type=int, default=0, help='Minimum number of users per item')
    parser.add_argument('--heldout_users', type=int, required=True, help='Number of held-out users for validation and testing')
    
    return parser.parse_args()

def get_count(tp, id_column):
    """
    Count the number of occurrences for each unique id in the specified column.

    Args:
        tp (pd.DataFrame): The DataFrame containing the data.
        id_column (str): The column name to count occurrences.

    Returns:
        pd.Series: A Series with counts indexed by the unique ids.
    """
    return tp.groupby(id_column).size()

def filter_triplets(tp, min_uc=5, min_sc=0):
    """
    Filter the triplet data based on minimum user and item counts.

    Args:
        tp (pd.DataFrame): The DataFrame containing user-item interactions.
        min_uc (int): Minimum number of items per user.
        min_sc (int): Minimum number of users per item.

    Returns:
        tuple: Filtered DataFrame, user counts, and item counts.
    """
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount[itemcount >= min_sc].index)]
    
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount[usercount >= min_uc].index)]
    
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.2, seed=98765):
    """
    Split the data into training and testing sets based on the specified proportion.

    Args:
        data (pd.DataFrame): The DataFrame to split.
        test_prop (float): Proportion of data to include in the test set.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Training DataFrame and testing DataFrame.
    """
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = [], []

    np.random.seed(seed)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)
        # print(n_items_u)
        if n_items_u >= 5:
            test_size = max(1, int(test_prop * n_items_u))
            test_indices = np.random.choice(n_items_u, size=test_size, replace=False)
            te_list.append(group.iloc[test_indices])
            tr_list.append(group.drop(group.index[test_indices]))
        else:
            tr_list.append(group)

        if (i + 1) % 1000 == 0:
            print(f"{i + 1} users sampled")
            sys.stdout.flush()
    # print(tr_list)
    data_tr = pd.concat(tr_list).reset_index(drop=True)
    data_te = pd.concat(te_list).reset_index(drop=True)
    
    return data_tr, data_te

def numerize(tp, profile2id, show2id):
    """
    Convert user and item IDs to numerical indices.

    Args:
        tp (pd.DataFrame): The DataFrame containing user-item interactions.
        profile2id (dict): Mapping from user IDs to numerical indices.
        show2id (dict): Mapping from item IDs to numerical indices.

    Returns:
        pd.DataFrame: DataFrame with numerical user and item indices.
    """
    uid = tp['userId'].map(profile2id)
    sid = tp['movieId'].map(show2id)
    return pd.DataFrame({'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def main():
    args = parse_args()

    dataset = args.dataset
    output_dir = args.output_dir
    threshold = args.threshold
    min_uc = args.min_items_per_user
    min_sc = args.min_users_per_item
    n_heldout_users = args.heldout_users

    # Load and filter raw data
    raw_data = pd.read_csv(dataset, header=0)
    raw_data = raw_data[raw_data['rating'] > threshold]
    print(f"Initial data shape: {raw_data.shape}")

    # Filter triplets
    raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc, min_sc)

    print(user_activity,item_popularity)
    # Compute Density and Sparsity
    density = raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    sparsity = 1.0 - density

    print(f"After filtering, there are {raw_data.shape[0]} watching events from {user_activity.shape[0]} users and {item_popularity.shape[0]} movies")
    print(f"Density: {density * 100:.3f}%")
    print(f"Sparsity: {sparsity * 100:.3f}%")

    unique_uid = user_activity.index.to_numpy()

    # Shuffle users
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]
    
    n_users = unique_uid.size
    
    # Split users into training, validation, and test sets
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]
    
    # Create training plays
    train_plays = raw_data[raw_data['userId'].isin(tr_users)]
    
    unique_sid = train_plays['movieId'].unique()
    
    show2id = {sid: i for i, sid in enumerate(unique_sid)}
    profile2id = {pid: i for i, pid in enumerate(unique_uid)}
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save unique show and user IDs
    with open(os.path.join(output_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write(f"{sid}\n")
            
    with open(os.path.join(output_dir, 'unique_uid.txt'), 'w') as f:
        for uid in unique_uid:
            f.write(f"{uid}\n")
    
    # Split validation and test plays
    vad_plays = raw_data[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays[vad_plays['movieId'].isin(unique_sid)]
    
    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
    
    test_plays = raw_data[raw_data['userId'].isin(te_users)]
    test_plays = test_plays[test_plays['movieId'].isin(unique_sid)]
    
    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)
    
    # Numerize and save datasets
    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    
    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(output_dir, 'validation_tr.csv'), index=False)
    
    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(output_dir, 'validation_te.csv'), index=False)
    
    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(output_dir, 'test_tr.csv'), index=False)
    
    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(output_dir, 'test_te.csv'), index=False)
    
    print("Data processing complete. Files saved to:", output_dir)


if __name__ == "__main__":
    main()
