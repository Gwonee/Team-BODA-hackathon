'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
'''
import numpy as np
import pandas as pd
from os.path import join, dirname
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

def create_missing_mask(data_tabular_path, mask_path, random_seed, missing_strategy, missing_rate):
    '''
    missing_strategy: value (random value missingness) or feature (random feature missingness)
    missing_rate: 0.0-1.0
    '''
    data_tabular = np.array(pd.read_csv(data_tabular_path, header=None))
    print(f'data tabular shape: {data_tabular.shape}')
    np.random.seed(random_seed)
    M, N = data_tabular.shape[0], data_tabular.shape[1]
    if missing_strategy == 'value':
        missing_mask_data = np.zeros((M*N), dtype=bool)
        mask_pos = np.random.choice(M*N, size=int(M*N*missing_rate), replace=False)
        missing_mask_data[mask_pos] = True
        missing_mask_data = missing_mask_data.reshape((M,N))
    elif missing_strategy == 'feature':
        missing_mask_data = np.zeros((M,N), dtype=bool)
        mask_pos = np.random.choice(N, size=int(N*missing_rate), replace=False)
        missing_mask_data[:,mask_pos] = True
    else:
        raise print('Only support value and feature missing strategy')
    np.save(mask_path, missing_mask_data)
    print(f'Real missing rate: {missing_mask_data.sum()/missing_mask_data.size}')
    print(f'Save missing mask to {mask_path}')
    return missing_mask_data

def create_certain_missing_mask(data_tabular_path, mask_path, mask_pos_order, missing_strategy, missing_rate):
    '''Create mask according to a mask order list (for MI and LI feature missingness)'''
    data_tabular = np.array(pd.read_csv(data_tabular_path, header=None))
    print(f'data tabular shape: {data_tabular.shape}')
    M, N = data_tabular.shape[0], data_tabular.shape[1]
    assert N == len(mask_pos_order)
    mask_pos = mask_pos_order[:int(N*missing_rate)]
    missing_mask_data = np.zeros((M,N), dtype=bool)
    missing_mask_data[:,mask_pos] = True
    np.save(mask_path, missing_mask_data)
    print(f'Real missing rate: {missing_mask_data.sum()/missing_mask_data.size}')
    print(f'Save missing mask to {mask_path}')
    return missing_mask_data

# TODO: change to your own path
FEATURES = '/bigdata/siyi/data/UKBB/cardiac_segmentations/projects/SelfSuperBio/18545/final'
MASK_PATH = join(FEATURES, 'missing_mask')

missing_strategy = 'value' 
missing_rate = 0.0

for target in ['CAD', 'Infarction']:
    train_name = 'cardiac_features_train_imputed_noOH_tabular_imaging_reordered.csv'
    val_name = 'cardiac_features_val_imputed_noOH_tabular_imaging_reordered.csv'
    test_name = 'cardiac_features_test_imputed_noOH_tabular_imaging_reordered.csv'
    for name, seed, split in zip([train_name, val_name, test_name], [2021,2022,2023], ['train', 'val', 'test']):
        save_mask_path = join(MASK_PATH, f'{name[:-4]}_{target}_{missing_strategy}_{missing_rate}.npy')
        path = join(FEATURES, name)
        # print(path)
        create_missing_mask(path, save_mask_path, seed, missing_strategy, missing_rate)

    balanced_train_name = f'cardiac_features_train_imputed_noOH_tabular_imaging_{target}_balanced_reordered.csv'
    balanced_path = join(FEATURES, balanced_train_name)
    balanced_save_mask_path = join(MASK_PATH, f'{balanced_train_name[:-4]}_{target}_{missing_strategy}_{missing_rate}.npy')
    create_missing_mask(balanced_path, balanced_save_mask_path, 2021, missing_strategy, missing_rate)

missing_strategy = 'feature'

for target in ['CAD', 'Infarction']:
    train_name = 'cardiac_features_train_imputed_noOH_tabular_imaging_reordered.csv'
    val_name = 'cardiac_features_val_imputed_noOH_tabular_imaging_reordered.csv'
    test_name = 'cardiac_features_test_imputed_noOH_tabular_imaging_reordered.csv'
    for name, seed, split in zip([train_name, val_name, test_name], [2022,2022,2022], ['train', 'val', 'test']):
        save_mask_path = join(MASK_PATH, f'{name[:-4]}_{target}_{missing_strategy}_{missing_rate}.npy')
        path = join(FEATURES, name)
        # print(path)
        create_missing_mask(path, save_mask_path, seed, missing_strategy, missing_rate)
    balanced_train_name = f'cardiac_features_train_imputed_noOH_tabular_imaging_{target}_balanced_reordered.csv'
    balanced_path = join(FEATURES, balanced_train_name)
    balanced_save_mask_path = join(MASK_PATH, f'{balanced_train_name[:-4]}_{target}_{missing_strategy}_{missing_rate}.npy')
    create_missing_mask(balanced_path, balanced_save_mask_path, 2022, missing_strategy, missing_rate)

# Check
train_np = np.load(join(MASK_PATH, f'{train_name[:-4]}_CAD_feature_0.3.npy'))
val_np = np.load(join(MASK_PATH, f'{val_name[:-4]}_CAD_feature_0.3.npy'))
test_np = np.load(join(MASK_PATH, f'{test_name[:-4]}_CAD_feature_0.3.npy'))
print(train_np[0])
print(val_np[0])
print(test_np[0])

target = 'CAD'
rf = RandomForestClassifier(random_state=2022)
# imbalanced
# X_train = pd.read_csv(join(FEATURES, 'cardiac_features_train_imputed_noOH_tabular_imaging_reordered.csv'), header=None)
# X_test = pd.read_csv(join(FEATURES, f'cardiac_features_test_imputed_noOH_tabular_imaging_reordered.csv'), header=None)
# y_train = torch.load(join(FEATURES, f'cardiac_labels_{target}_train.pt'))
# y_test = torch.load(join(FEATURES, f'cardiac_labels_{target}_test.pt'))

# balanced
X_train = pd.read_csv(join(FEATURES, f'cardiac_features_train_imputed_noOH_tabular_imaging_{target}_balanced_reordered.csv'), header=None)
X_test = pd.read_csv(join(FEATURES, f'cardiac_features_test_imputed_noOH_tabular_imaging_reordered.csv'), header=None)
y_train = torch.load(join(FEATURES, f'cardiac_labels_{target}_train_balanced.pt'))
y_test = torch.load(join(FEATURES, f'cardiac_labels_{target}_test.pt'))
rf.fit(X_train, y_train)
# Predict probabilities for the test dataset
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)

print(f"AUC on test dataset: {auc}")

data_df = pd.read_csv(join(dirname(FEATURES), f'cardiac_feature_18545_vector_labeled_noOH_dropNI_imputed.csv'),nrows=5)
field_lengths_tabular = torch.load(join(dirname(FEATURES), 'tabular_lengths.pt'))
categorical_ids = []
continuous_ids = []
for i in range(len(field_lengths_tabular)):
    if field_lengths_tabular[i] == 1:
        continuous_ids.append(i)
    else:
        categorical_ids.append(i)
column_name = data_df.columns[1:]
column_name = column_name[categorical_ids+continuous_ids]
# print(column_name)

# Get feature importances
importances = rf.feature_importances_
# Sort feature importances in descending order
MI_indices = np.argsort(importances)[::-1]
LI_indices = np.argsort(importances)
# Get feature names
MI_feature_name = column_name[MI_indices]
print(MI_feature_name)

missing_rate = 0.9
missing_strategy = 'MI'

train_name = 'cardiac_features_train_imputed_noOH_tabular_imaging_reordered.csv'
val_name = 'cardiac_features_val_imputed_noOH_tabular_imaging_reordered.csv'
test_name = 'cardiac_features_test_imputed_noOH_tabular_imaging_reordered.csv'
for name, split in zip([train_name, val_name, test_name], ['train', 'val', 'test']):
    save_mask_path = join(MASK_PATH, f'{name[:-4]}_{target}_{missing_strategy}_{missing_rate}.npy')
    path = join(FEATURES, name)
    # print(path)
    create_certain_missing_mask(path, save_mask_path, MI_indices, missing_strategy, missing_rate)

balanced_train_name = f'cardiac_features_train_imputed_noOH_tabular_imaging_{target}_balanced_reordered.csv'
balanced_path = join(FEATURES, balanced_train_name)
balanced_save_mask_path = join(MASK_PATH, f'{balanced_train_name[:-4]}_{target}_{missing_strategy}_{missing_rate}.npy')
create_certain_missing_mask(balanced_path, balanced_save_mask_path, MI_indices, missing_strategy, missing_rate)

missing_strategy = 'LI'
train_name = 'cardiac_features_train_imputed_noOH_tabular_imaging_reordered.csv'
val_name = 'cardiac_features_val_imputed_noOH_tabular_imaging_reordered.csv'
test_name = 'cardiac_features_test_imputed_noOH_tabular_imaging_reordered.csv'
for name, split in zip([train_name, val_name, test_name], ['train', 'val', 'test']):
    save_mask_path = join(MASK_PATH, f'{name[:-4]}_{target}_{missing_strategy}_{missing_rate}.npy')
    path = join(FEATURES, name)
    # print(path)
    create_certain_missing_mask(path, save_mask_path, LI_indices, missing_strategy, missing_rate)

balanced_train_name = f'cardiac_features_train_imputed_noOH_tabular_imaging_{target}_balanced_reordered.csv'
balanced_path = join(FEATURES, balanced_train_name)
balanced_save_mask_path = join(MASK_PATH, f'{balanced_train_name[:-4]}_{target}_{missing_strategy}_{missing_rate}.npy')
result = create_certain_missing_mask(balanced_path, balanced_save_mask_path, LI_indices, missing_strategy, missing_rate)

# Check train, val, test to miss the same columns
train_np = np.load(join(MASK_PATH, 'cardiac_features_train_imputed_noOH_tabular_imaging_CAD_balanced_reordered_CAD_MI_0.9.npy'))
val_np = np.load(join(MASK_PATH, 'cardiac_features_val_imputed_noOH_tabular_imaging_reordered_CAD_MI_0.9.npy'))
test_np = np.load(join(MASK_PATH, 'cardiac_features_test_imputed_noOH_tabular_imaging_reordered_CAD_MI_0.9.npy'))
print(np.where(train_np[0]))
print(np.where(val_np[0]))
print(np.where(test_np[0]))

# TODO: change to your own path
FEATURES = '/bigdata/siyi/data/DVM/features'
MASK_PATH = join(FEATURES, 'missing_mask')

missing_strategy = 'value'
missing_rate = 0.0
target = 'dvm'

train_name = 'dvm_features_train_noOH_all_views_physical_jittered_50_reordered.csv'
val_name = 'dvm_features_val_noOH_all_views_physical_jittered_50_reordered.csv'
test_name = 'dvm_features_test_noOH_all_views_physical_jittered_50_reordered.csv'
for name, seed, split in zip([train_name, val_name, test_name], [2021,2022,2023], ['train', 'val', 'test']):
    save_mask_path = join(MASK_PATH, f'{name[:-4]}_{target}_{missing_strategy}_{missing_rate}.npy')
    path = join(FEATURES, name)
    # print(path)
    create_missing_mask(path, save_mask_path, seed, missing_strategy, missing_rate)

missing_strategy = 'feature'

train_name = 'dvm_features_train_noOH_all_views_physical_jittered_50_reordered.csv'
val_name = 'dvm_features_val_noOH_all_views_physical_jittered_50_reordered.csv'
test_name = 'dvm_features_test_noOH_all_views_physical_jittered_50_reordered.csv'
for name, seed, split in zip([train_name, val_name, test_name], [2022,2022,2022], ['train', 'val', 'test']):
    save_mask_path = join(MASK_PATH, f'{name[:-4]}_{target}_{missing_strategy}_{missing_rate}.npy')
    path = join(FEATURES, name)
    # print(path)
    create_missing_mask(path, save_mask_path, seed, missing_strategy, missing_rate)

# Check train, val, test to miss the same columns
train_np = np.load(join(MASK_PATH, f'{train_name[:-4]}_dvm_feature_0.3.npy'))
val_np = np.load(join(MASK_PATH, f'{val_name[:-4]}_dvm_feature_0.3.npy'))
test_np = np.load(join(MASK_PATH, f'{test_name[:-4]}_dvm_feature_0.3.npy'))
print(train_np[0])
print(val_np[0])
print(test_np[0])

from sklearn.metrics import accuracy_score
rf = RandomForestClassifier(random_state=2022)
X_train = pd.read_csv(join(FEATURES, 'dvm_features_train_noOH_all_views_physical_jittered_50_reordered.csv'), header=None)
X_test = pd.read_csv(join(FEATURES, f'dvm_features_test_noOH_all_views_physical_jittered_50_reordered.csv'), header=None)
y_train = torch.load(join(FEATURES, 'labels_model_all_train_all_views.pt'))
y_test = torch.load(join(FEATURES, 'labels_model_all_test_all_views.pt'))
rf.fit(X_train, y_train)

# Predict classes for the test dataset
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on test dataset: {accuracy}")

reordered_column_name = ['Color', 'Bodytype', 'Gearbox','Fuel_type' ,
               'Wheelbase', 'Height', 'Width', 'Length', 'Adv_year', 'Adv_month',
       'Reg_year', 'Runned_Miles', 'Price', 'Seat_num', 'Door_num',
       'Entry_price', 'Engine_size']

# Get feature importances
importances = rf.feature_importances_
# Sort feature importances in descending order
MI_indices = np.argsort(importances)[::-1]
LI_indices = np.argsort(importances)
# Get feature names
MI_feature_name = [reordered_column_name[x] for x in MI_indices]
print(MI_feature_name)

missing_rate = 0.1

missing_strategy = 'MI'
train_name = 'dvm_features_train_noOH_all_views_physical_jittered_50_reordered.csv'
val_name = 'dvm_features_val_noOH_all_views_physical_jittered_50_reordered.csv'
test_name = 'dvm_features_test_noOH_all_views_physical_jittered_50_reordered.csv'
for name, split in zip([train_name, val_name, test_name], ['train', 'val', 'test']):
    save_mask_path = join(MASK_PATH, f'{name[:-4]}_{target}_{missing_strategy}_{missing_rate}.npy')
    path = join(FEATURES, name)
    create_certain_missing_mask(path, save_mask_path, MI_indices, missing_strategy, missing_rate)

missing_strategy = 'LI'
train_name = 'dvm_features_train_noOH_all_views_physical_jittered_50_reordered.csv'
val_name = 'dvm_features_val_noOH_all_views_physical_jittered_50_reordered.csv'
test_name = 'dvm_features_test_noOH_all_views_physical_jittered_50_reordered.csv'
for name, split in zip([train_name, val_name, test_name], ['train', 'val', 'test']):
    save_mask_path = join(MASK_PATH, f'{name[:-4]}_{target}_{missing_strategy}_{missing_rate}.npy')
    path = join(FEATURES, name)
    create_certain_missing_mask(path, save_mask_path, LI_indices, missing_strategy, missing_rate)

train_np = np.load(join(MASK_PATH, f'{train_name[:-4]}_dvm_MI_0.3.npy'))
val_np = np.load(join(MASK_PATH, f'{val_name[:-4]}_dvm_MI_0.3.npy'))
test_np = np.load(join(MASK_PATH, f'{test_name[:-4]}_dvm_MI_0.3.npy'))
print(train_np[0])
print(val_np[0])
print(test_np[0])