from xgboost import XGBRegressor
from charge_density_dataset import ChargeDensityDataset
import numpy as np

import os
from tqdm import tqdm
import time

clf = XGBRegressor(tree_method='gpu_hist', gpu_id=0, max_depth=10)

data = np.load('/data/therealgabeguo/crystallography/charge_data_npy/test/CHGCAR_mp-1920.npy')
print(data.shape)

train_features = list()
train_outputs = list()

# test_features = list()
# test_outputs = list()

for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        for z in range(data.shape[2]):
            if x % 2 == 0 and y % 2 == 0 and z % 2 == 0:
                train_features.append([x, y, z])
                train_outputs.append(data[x, y, z])
            # else:
            #     test_features.append([x, y, z])
            #     test_outputs.append(data[x, y, z])

train_features = np.array(train_features)
train_outputs = np.array(train_outputs)

print('training')
time_start = time.time()
clf.fit(train_features, train_outputs)
time_end = time.time()
print('training took {:.2f} seconds'.format(time_end - time_start))

print('predicting')
the_output = np.zeros_like(data)
pred_coords = list()
time_start = time.time()
for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        for z in range(data.shape[2]):
            pred_coords.append([x, y, z])
pred_coords = np.array(pred_coords)
pred_outputs = clf.predict(pred_coords)
time_end = time.time()
print('prediction took {:.2f} seconds'.format(time_end - time_start))
for i in range(len(pred_coords)):
    the_output[int(pred_coords[i][0]), int(pred_coords[i][1]), int(pred_coords[i][2])] = pred_outputs[i]

np.save('xgboost_mp-1920.npy', the_output)
clf.save_model('xgboost_mp-1920.model')