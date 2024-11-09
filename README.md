# CAM-Transformer
Applying different Channel Attention Modules (CAMs) on Transformer-based model for 3D hypercloud segmentation in a geological application. Find our paper [here](https://isprs-archives.copernicus.org/articles/XLVIII-4-W11-2024/103/2024/).

<img width="500" alt="Screenshot 2024-11-09 at 02 49 55" src="https://github.com/user-attachments/assets/3abbc984-d919-424d-9971-f0f1406ef6e6">

<img width="800" alt="Screenshot 2024-11-09 at 02 51 50" src="https://github.com/user-attachments/assets/d88433bb-e308-459c-b571-7637b6aaeed0">

<img width="300" alt="Screenshot 2024-11-09 at 02 51 57" src="https://github.com/user-attachments/assets/98b462de-8849-4937-8f5f-99ef04d20ddb">

Download Tinto data from [RODARE](https://rodare.hzdr.de/record/2256).

First prepare KNN points:
```
python find_kNN_for_training.py
python find_kNN_for_testing.py
```
Then train and test all models:
```
Train_and_Test_CAM-Transformer.sh
```
Visualize Channel Attention:
```
python Visualize_CAM.py
```

Cite the paper here:
>Rizaldy, A., Ghamisi, P., and Gloaguen, R.: Channel Attention Module for Segmentation of 3D Hyperspectral Point Clouds in Geological Applications, Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLVIII-4/W11-2024, 103–109, https://doi.org/10.5194/isprs-archives-XLVIII-4-W11-2024-103-2024, 2024.
