import numpy as np
import torch
import torch.nn as nn

from model import SE_PT_PCTCls, ECA_PT_PCTCls, CBAM_PT_PCTCls, CAM_PT_PCTCls

device = 'cuda'

model_path = 'checkpoint/SE_vnir_degr_pt_pct_100epochs/model-best.pth'
model = SE_PT_PCTCls(51).to(device)
model = nn.DataParallel(model) 
model.load_state_dict(torch.load(model_path))
model = model.eval()

pc = np.load('data/test_vnir_degr_knn/test_vnir_degr_knn0.npy')[:10,:,:-1]
pc = torch.from_numpy(pc).permute(0,2,1)

out = model(pc.float())

