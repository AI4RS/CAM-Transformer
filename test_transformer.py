# code based on: https://github.com/qinglew/PointCloudTransformer
# cls.py

import argparse
import os
import time

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import PointData, PointData_Test, n_dim_count
from model import NaivePCTCls, SPCTCls, ASCN_PCTCls, PT_PCTCls, CAM_PT_PCTCls, SE_PT_PCTCls, ECA_PT_PCTCls, CBAM_PT_PCTCls, mCAM_PT_PCTCls
from util import cal_loss, Logger, plot_tr


models = {'naive_pct': NaivePCTCls,
          'spct': SPCTCls,
          'ascn_pct': ASCN_PCTCls,
          'pt_pct': PT_PCTCls,
          'cam_pt_pct': CAM_PT_PCTCls,
          'se_pt_pct': SE_PT_PCTCls,
          'eca_pt_pct': ECA_PT_PCTCls,
          'cbam_pt_pct': CBAM_PT_PCTCls,
          'mcam_pt_pct': mCAM_PT_PCTCls}
        
def _init_(args):
    if not os.path.exists('predict/'+args.exp_name):
        os.makedirs('predict/'+args.exp_name)

def test(args,io):
    data_test_list = os.listdir(args.data_test_folder)
    for i in range(len(data_test_list)):
        args.data_test = os.path.join(args.data_test_folder,data_test_list[i])
        print('Test for ', args.data_test)
        # test()
    
        test_loader = DataLoader(PointData_Test(path=args.data_test),
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False)

        device = torch.device("cuda" if args.cuda else "cpu")

        n_dim = n_dim_count(args.data_test)

        model = models[args.model](n_dim=n_dim).to(device)
        model = nn.DataParallel(model) 

        model.load_state_dict(torch.load(args.model_path))
        model = model.eval()

        test_true = []
        test_pred = []

        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = model(data)
            preds = logits.max(dim=1)[1] 
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
        print('Test acc = ', test_acc)
        io.cprint(outstr)

        test_true_pred = np.zeros((test_true.shape[0],2))
        test_true_pred[:,0] = test_true
        test_true_pred[:,1] = test_pred
        np.savetxt('predict/'+args.exp_name+'/'+args.data_test[-19:-4]+'.txt', test_true_pred, fmt='%1.0f %1.0f')

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pct', choices=['naive_pct', 'spct', 'pct', 'ascn_pct', 'pt_pct', 'cam_pt_pct', 'se_pt_pct', 'eca_pt_pct', 'cbam_pt_pct', 'mcam_pt_pct'],
                        help='which model you want to use')
    parser.add_argument('--data_train', type=str, default='', metavar='N',
                        help='Path of the training data')
    parser.add_argument('--data_test_folder', type=str, default='', metavar='N',
                        help='Path of the testing data')
    parser.add_argument('--data_test', type=str, default='', metavar='N',
                        help='Path of the testing data')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='enables CUDA training')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    
    _init_(args)
    
    io = Logger('predict/' + args.exp_name + '/run_test.log')
    io.cprint(str(args))
    
    if args.cuda:
        io.cprint('Train with CUDA')
    
    test(args, io)
