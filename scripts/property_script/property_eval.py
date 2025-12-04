import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


if __name__ == '__main__':

    root_path = f"./results/property_result"
    
    data_list = ["Egc", "Egb", "Eea", "Ei", "Xc", "EPS", "Nc", "Eat"]
    fold_list = [1, 2, 3, 4, 5]

    for data in data_list:
        
        print(f'*********************************data-{data}*********************************')
        
        rmse_list = []
        r2_list = []
        
        for fold in fold_list:
            predict_path = os.path.join(root_path, f'result_{data}', f'result_{data}_fold_{fold}',  f'test.out.pkl')
            predict_outputs = pd.read_pickle(predict_path)

            target_list = []
            pred_list = []
            
            for epoch in range(len(predict_outputs)):
                predict_output = predict_outputs[epoch]
                target_list.append(predict_output['target'])
                pred_list.append(predict_output['predict'])
            
            target = torch.cat(target_list, dim=0).float()
            pred = torch.cat(pred_list, dim=0).float()
            
            y_true = target.cpu().numpy()
            y_pred = pred.cpu().numpy()
                        
            rmse = np.sqrt(np.mean(np.square(y_pred - y_true)))
            r2 = r2_score(y_true, y_pred)
                        
            rmse_list.append(rmse)
            r2_list.append(r2)

        print("RMSE: {:.3f} ± {:.3f}".format(np.mean(np.array(rmse_list)), np.std(np.array(rmse_list))))
        print("R2: {:.3f} ± {:.3f}".format(np.mean(np.array(r2_list)), np.std(np.array(r2_list))))
        
    
    
    
    
    

    
    