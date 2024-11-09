import numpy as np
from sklearn.neighbors import KDTree

data_type = 'degr' # choose: real, degr, synth
band = 'vnir' # choose: lwir, swir, vnir, all_HS
k = 20 # how many k-NN points?

if data_type == 'real':
    train_data_path = 'data/train_data_real.npy'
elif data_type == 'degr':
    train_data_path = 'data/train_data_degr.npy'
elif data_type == 'synth':
    train_data_path = 'data/train_data_synth.npy'

train_data = np.load(train_data_path)

# Check dimension
# XYZ + LWIR (126) + SWIR (141) + VNIR (51) + label = 322
print("train data", train_data.shape)

train_features = train_data[:,:-1]

train_label = train_data[:,-1]

print("train features", train_features.shape)
print("train label", train_label.shape)
print("train label = ", np.unique(train_label))

print(np.max(train_features[:,3:129]))
print(np.max(train_features[:,129:270]))
print(np.max(train_features[:,270:321]))

# for train set of LWIR, SWIR, VNIR
train_lwir = train_features[:,3:129]
train_swir = train_features[:,129:270]
train_vnir = train_features[:,270:321]
train_all_HS = train_features[:,3:321]
xyz = train_features[:,:3]

train_lwir_label = np.hstack((train_lwir, train_label[:,None]))
train_swir_label = np.hstack((train_swir, train_label[:,None]))
train_vnir_label = np.hstack((train_vnir, train_label[:,None]))
train_all_HS_label = np.hstack((train_all_HS, train_label[:,None]))

# k-NN indexing

tree = KDTree(xyz, leaf_size=2)              
dist, ind = tree.query(xyz, k=20)

print("... k-NN indexing finished")

# which bands?
if band == 'lwir':
    which_bands = train_lwir_label
elif band == 'swir':
    which_bands = train_swir_label
elif band == 'vnir':
    which_bands = train_vnir_label
elif band == 'all_HS':
    which_bands = train_all_HS_label

n_points = which_bands.shape[0]
dim = which_bands.shape[1]
print(dim)

# k-NN searching for train data (no need to split)
knn_results = np.zeros((n_points,k, dim))
for i in range(n_points): 
    knn_results[i,:,:] = which_bands[ind[i,:]]
path_name = 'data/train_'+band+'_'+data_type+'_knn.npy'
np.save(path_name, knn_results)
print('save ', path_name)
