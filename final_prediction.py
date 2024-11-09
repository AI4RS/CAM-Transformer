import numpy as np
import sklearn.metrics as metrics
import os
import argparse

def merge_all(args):
    pred_test_list = os.listdir(args.pred_test_folder)
    pred_test_all = []
    for file in pred_test_list:
        if file.endswith('.txt'):
            pred_test_file = os.path.join(args.pred_test_folder,file)
            pred_test = np.loadtxt(pred_test_file)
            pred_test_all.append(pred_test)

    pred_test_np_all = np.concatenate(pred_test_all)
    test_true = pred_test_np_all[:,0]
    test_pred = pred_test_np_all[:,1]
    print('prediction all shape', pred_test_np_all.shape)
    OA = np.array([metrics.accuracy_score(test_true, test_pred)])
    print('Overall accuracy = ', OA)
    
    matrix = metrics.confusion_matrix(test_true, test_pred)
    AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(test_true))
    print('Average accuracy = ', AA) 

    pred_all_filename = 'prediction_all.txt'
    np.savetxt(os.path.join(args.pred_test_folder,pred_all_filename), pred_test_np_all)
    print('Save all prediction')
    
    pred_acc_filename = 'accuracy.txt'
    OAAA = np.vstack((OA, AA))
    np.savetxt(os.path.join(args.pred_test_folder,pred_acc_filename), OAAA)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction All')
    parser.add_argument('--pred_test_folder', type=str, default='', metavar='N',
                        help='Path of the prediction')
    args = parser.parse_args()
    merge_all(args)
