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
from model import NaivePCTCls, SPCTCls, ASCN_PCTCls, PT_PCTCls, HPF_PT_PCTCls, CAM_PT_PCTCls, SE_PT_PCTCls, ECA_PT_PCTCls, CBAM_PT_PCTCls, mCAM_PT_PCTCls
from util import cal_loss, Logger, EarlyStopper, plot_tr


models = {'naive_pct': NaivePCTCls,
          'spct': SPCTCls,
          'ascn_pct': ASCN_PCTCls,
          'pt_pct': PT_PCTCls,
          'hpf_pt_pct': HPF_PT_PCTCls,
          'cam_pt_pct': CAM_PT_PCTCls,
          'se_pt_pct': SE_PT_PCTCls,
          'eca_pt_pct': ECA_PT_PCTCls,
          'cbam_pt_pct': CBAM_PT_PCTCls,
          'mcam_pt_pct': mCAM_PT_PCTCls}


def _init_(args):
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    if not os.path.exists('checkpoint/' + args.exp_name):
        os.makedirs('checkpoint/' + args.exp_name)


def train(args, io):
    train_loader = DataLoader(PointData(split='train', path=args.data_train), num_workers=2,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(PointData(split='val', path=args.data_train), num_workers=2,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda" if args.cuda else "cpu")

    n_dim = n_dim_count(args.data_train)
    io.cprint('number of dim = '+ str(n_dim))

    model = models[args.model](n_dim=n_dim).to(device)
    # print(model)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    # Add weight manually
    weights = torch.Tensor([0.9051251, 0.92573627, 0.95601125, 0.85206365, 0.98283415, 0.72755554, 0.78923368, 0.93059351, 0.97715274, 0.9536941])
    weights = weights.cuda()
    
    criterion = cal_loss
    
    best_test_acc = 0
    # if args.early_stop:
    #     outstr = 'Use Early Stop'
    #     early_stopper = EarlyStopper(patience=10, min_delta=10)
    
    plot_train = np.zeros((args.epochs,2))
    plot_test = np.zeros((args.epochs,2))

    for epoch in range(args.epochs):
        print('Epoch = ', epoch)
        train_loss = 0.0
        count = 0.0  # numbers of data
        model.train()
        train_pred = []
        train_true = []
        idx = 0  # iterations
        total_time = 0.0
        for data, label in (train_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            start_time = time.time()
            logits = model(data)
            # loss = criterion(logits, label) 
            loss = criterion(logits, label, weights, smoothing=False)
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1
            
        print ('train total time is',total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        plot_train[epoch,0] = train_loss*1.0/count
        plot_train[epoch,1] = metrics.accuracy_score(train_true, train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            start_time = time.time()
            logits = model(data)
            end_time = time.time()
            total_time += (end_time - start_time)
            # loss = criterion(logits, label)
            loss = criterion(logits, label, weights, smoothing=False)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        print ('test total time is', total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        plot_test[epoch,0] = test_loss*1.0/count
        plot_test[epoch,1] = metrics.accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                            test_loss*1.0/count,
                                                                            test_acc,
                                                                            avg_per_class_acc)
        io.cprint(outstr)
        torch.save(model.state_dict(), 'checkpoint/%s/epoch-%s.pth' % (args.exp_name, str(epoch)))
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoint/%s/model-best.pth' % (args.exp_name))
        
        scheduler.step()
        
        plot_tr(plot_train=plot_train, plot_test=plot_test, path='checkpoint/'+ args.exp_name)
        
        # if args.early_stop:
        #     if early_stopper.early_stop(test_loss):             
        #         outstr = 'Done with Early Stop !!'
        #         io.cprint(outstr)
        #         break

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pct', choices=['naive_pct', 'spct', 'pct', 'ascn_pct', 'pt_pct', 'hpf_pt_pct', 'cam_pt_pct', 'se_pt_pct', 'eca_pt_pct', 'cbam_pt_pct', 'mcam_pt_pct'],
                        help='which model you want to use')
    parser.add_argument('--data_train', type=str, default='', metavar='N',
                        help='Path of the training data')
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
    parser.add_argument('--early_stop', type=bool, default=False,
                       help='Early Stop')
    args = parser.parse_args()

    _init_(args)
    
    io = Logger('checkpoint/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    
    if args.cuda:
        io.cprint('Train with CUDA')
    # if args.early_stop:
    #     io.cprint('Train with early stopper')
    # else:
    #     io.cprint('Train without early stopper')
    
    train(args, io)
