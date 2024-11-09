# FOR TRAINING

# CAM DANET

# train with LWIR (standardscaler dataset)
python train_transformer.py --model 'cam_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_lwir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'CAM_lwir_degr_pt_pct_100epochs' \
                             --early_stop=False

# train with SWIR (standardscaler dataset)
python train_transformer.py --model 'cam_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_swir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'CAM_swir_degr_pt_pct_100epochs' \
                             --early_stop=False

# train with VNIR (standardscaler dataset)
python train_transformer.py --model 'cam_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_vnir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'CAM_vnir_degr_pt_pct_100epochs' \
                             --early_stop=False

# mCAM

# train with LWIR (standardscaler dataset)
python train_transformer.py --model 'mcam_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_lwir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'mCAM_lwir_degr_pt_pct_100epochs' \
                             --early_stop=False

# train with SWIR (standardscaler dataset)
python train_transformer.py --model 'mcam_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_swir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'mCAM_swir_degr_pt_pct_100epochs' \
                             --early_stop=False

# train with VNIR (standardscaler dataset)
python train_transformer.py --model 'mcam_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_vnir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'mCAM_vnir_degr_pt_pct_100epochs' \
                             --early_stop=False

# SENet

# train with LWIR (standardscaler dataset)
python train_transformer.py --model 'se_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_lwir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'SE_lwir_degr_pt_pct_100epochs' \
                             --early_stop=False

# train with SWIR (standardscaler dataset)
python train_transformer.py --model 'se_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_swir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'SE_swir_degr_pt_pct_100epochs' \
                             --early_stop=False

# train with VNIR (standardscaler dataset)
python train_transformer.py --model 'se_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_vnir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'SE_vnir_degr_pt_pct_100epochs' \
                             --early_stop=False

# ECANet

# train with LWIR (standardscaler dataset)
python train_transformer.py --model 'eca_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_lwir_degr_knn.npy' \
                             --epochs 5 \
                             --exp_name 'ECA_lwir_degr_pt_pct_100epochs' \
                             --early_stop=False

# train with SWIR (standardscaler dataset)
python train_transformer.py --model 'eca_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_swir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'ECA_swir_degr_pt_pct_100epochs' \
                             --early_stop=False

# train with VNIR (standardscaler dataset)
python train_transformer.py --model 'eca_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_vnir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'ECA_vnir_degr_pt_pct_100epochs' \
                             --early_stop=False

# CBAM

# train with LWIR (standardscaler dataset)
python train_transformer.py --model 'cbam_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_lwir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'CBAM_lwir_degr_pt_pct_100epochs' \
                             --early_stop=False

# train with SWIR (standardscaler dataset)
python train_transformer.py --model 'cbam_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_swir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'CBAM_swir_degr_pt_pct_100epochs' \
                             --early_stop=False

# train with VNIR (standardscaler dataset)
python train_transformer.py --model 'cbam_pt_pct' \
                             --lr 0.00001 \
                             --data_train 'data/REtrain_vnir_degr_knn.npy' \
                             --epochs 100 \
                             --exp_name 'CBAM_vnir_degr_pt_pct_100epochs' \
                             --early_stop=False

# FOR TESTING

# CAM DANET

# test with LWIR (standardscaler dataset)
python test_transformer.py --model 'cam_pt_pct' \
                            --data_test_folder 'data/test_lwir_degr_knn/' \
                            --model_path 'checkpoint/CAM_lwir_degr_pt_pct_100epochs/model-best.pth' \
                            --exp_name 'CAM_test_lwir_degr_pt_pct_100epochs'

# MERGE TESTING RESULTS

# CAM DANET

python final_prediction.py --pred_test_folder 'predict/CAM_test_lwir_degr_pt_pct_100epochs'