import numpy as np
from sklearn.neighbors import KDTree

data_type = 'degr' # choose: real, degr, synth
band = 'vnir' # choose: lwir, swir, vnir, all_HS
k = 20 # how many k-NN points?
n_points_temp = 250000 # how many points for splitting testing data

if data_type == 'real':
    test_data_path = 'data/test_data_real.npy'
elif data_type == 'degr':
    test_data_path = 'data/test_data_degr.npy'
elif data_type == 'synth':
    test_data_path = 'data/test_data_synth.npy'

test_data = np.load(test_data_path)

# Check dimension
# XYZ + LWIR (126) + SWIR (141) + VNIR (51) + label = 322
print("test data", test_data.shape)

test_features = test_data[:,:-1]

test_label = test_data[:,-1]

print("test features", test_features.shape)
print("test label", test_label.shape)
print("test label = ", np.unique(test_label))


print(np.max(test_features[:,3:129]))
print(np.max(test_features[:,129:270]))
print(np.max(test_features[:,270:321]))

# for test set of LWIR, SWIR, VNIR
test_lwir = test_features[:,3:129]
test_swir = test_features[:,129:270]
test_vnir = test_features[:,270:321]
test_all_HS = test_features[:,3:321]
xyz = test_features[:,:3]

test_lwir_label = np.hstack((test_lwir, test_label[:,None]))
test_swir_label = np.hstack((test_swir, test_label[:,None]))
test_vnir_label = np.hstack((test_vnir, test_label[:,None]))
test_all_HS_label = np.hstack((test_all_HS, test_label[:,None]))

# k-NN indexing

tree = KDTree(xyz, leaf_size=2)              
dist, ind = tree.query(xyz, k=20)

print("... k-NN indexing finished")

# which bands?
if band == 'lwir':
    which_bands = test_lwir_label
elif band == 'swir':
    which_bands = test_swir_label
elif band == 'vnir':
    which_bands = test_vnir_label
elif band == 'all_HS':
    which_bands = test_all_HS_label

n_points = which_bands.shape[0]
dim = which_bands.shape[1]
print(dim)

# k-NN searching for test data (splitted and repeated automatically)
import os
os.makedirs('data/test_'+band+'_'+data_type+'_knn/')

n_iter = int(n_points / n_points_temp)
start = 0
for n in range(n_iter):
    knn_results = np.zeros((n_points_temp,k, dim))
    end = start + n_points_temp
    for i in range(start,end): 
        knn_results[i-start,:,:] = which_bands[ind[i,:]]
    path_name = 'data/test_'+band+'_'+data_type+'_knn/test_'+band+'_'+data_type+'_knn'+str(n)+'.npy'
    np.save(path_name, knn_results)
    print('save ', path_name)
    start += n_points_temp

knn_results = np.zeros((n_points - (n_iter*n_points_temp),k, dim))
start = n_iter*n_points_temp
end = n_points
for i in range(start,end): 
    knn_results[i-start,:,:] = which_bands[ind[i,:]]
n = n_iter+1
path_name = 'data/test_'+band+'_'+data_type+'_knn/test_'+band+'_'+data_type+'_knn'+str(n)+'.npy'
np.save(path_name, knn_results)
print('save ', path_name)
start += n_points_temp
