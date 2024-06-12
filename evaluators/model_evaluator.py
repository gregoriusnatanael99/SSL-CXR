import numpy as np
import pandas as pd

import torch
from torch import nn
from misc.utils import *

# from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle

class Model_Evaluator():
    def __init__(self,cfg_data,dataloaders,dataset_sizes):
        self.cfg_data = cfg_data
        self.cfg_data['TL_ALGO'] = ''
        self.dataloaders = dataloaders
        self.device = torch.device("cuda:"+str(cfg_data.GPU_ID[0]) if torch.cuda.is_available() else "cpu")
        self.dataset_sizes = dataset_sizes
        self.model = self.initialize_model()
        self.model.load_state_dict(torch.load(os.path.join(self.cfg_data.MODEL_DIR,self.cfg_data.MODEL_NAME)))

    def begin_testing(self):
        criterion = nn.CrossEntropyLoss()

        test_loss = []
        y_pred, y_true, = [], []
        y_prob = []

        confusion_matrix = torch.zeros(self.cfg_data.num_class, self.cfg_data.num_class)
        count = 0
        self.model.to(self.device)
        print("Evaluating model . . .")
        with torch.no_grad():
            for inputs, classes in self.dataloaders:
#                print(inputs.shape)
                inputs = inputs.to(self.device)
                labels = classes.to(self.device)

                outputs = self.model(inputs)
                print(outputs)
                _, preds = torch.max(outputs, 1)
#                preds = np.argmax(outputs.cpu().tolist())
                print(preds,labels.cpu().tolist()[0])
                test_loss.append(criterion(outputs, labels).cpu().numpy())

                count+=1
                print("Processed {} images, current loss = {}".format(count,np.mean(test_loss)))
    #             print(preds,labels)
                confusion_matrix[labels.cpu().tolist()[0], preds.cpu().tolist()[0]] += 1
                y_pred.append(preds.cpu().tolist())
                y_true.append(labels.cpu().tolist())
                y_prob.append(outputs.cpu().tolist()[0])
                # print(y_prob)
        print("Building confusion matrix . . .")
        res_df = export_conf_mat(confusion_matrix,class_names=self.cfg_data.class_names,exp_path=self.cfg_data.MODEL_DIR)

        print("Generating ROC . . .")
        roc_auc = self.calculate_roc(y_true,y_pred,y_prob,class_names=self.cfg_data.class_names,exp_path=self.cfg_data.MODEL_DIR,save_bool=True)

        print("Calculating evaluation metrics . . .")
        metric_df = self.calculate_classification_metrics_from_conf_mat(res_df,self.cfg_data.class_names,roc_auc)
        res_dict = {}
        res_dict['loss'] = np.mean(test_loss)
        res_dict['acc'] = np.sum(metric_df['tp'])/np.sum(confusion_matrix.numpy().astype('int'))
        # roc_dict = self.calc_roc_curve(y_test=y_true,y_score=y_pred,n_classes=self.cfg_data['num_class'])

        res_dict['macro_auc'] = roc_auc['macro']
        # res_dict['micro_auc'] = roc_dict['micro']

        # macro
        res_dict['macro_prec'] = np.mean(metric_df['precision'])
        res_dict['macro_rec'] = np.mean(metric_df['recall'])
        res_dict['macro_f1'] = np.mean(metric_df['F1'])

        # micro
        # res_dict['micro_prec'] = np.sum(metric_df['tp'])/np.sum(metric_df['tp'])+np.sum(metric_df['fp'])
        # res_dict['micro_rec'] = np.sum(metric_df['tp'])/np.sum(metric_df['tp'])+np.sum(metric_df['fn'])
        # res_dict['micro_f1'] = 2*((res_dict['micro_prec']*res_dict['micro_rec'])/(res_dict['micro_prec']+res_dict['micro_rec']))

        res_df.to_csv(os.path.join(self.cfg_data.MODEL_DIR,"confusion_matrix.csv"))

        log_test_data(self.cfg_data.MODEL_DIR,res_dict,metric_df)

    def calculate_classification_metrics_from_conf_mat(self,res_df,class_names,roc_auc):
        res_dict = {'tp':{},'fp':{},'fn':{},'tn':{},'fpr':{},'fnr':{},'tpr':{},'tnr':{},'precision':{}
                    ,'recall':{},'F1':{},'roc_auc':{}}
        for i in class_names:
            tp = res_df['pred_'+i]['gt_'+i]
            fn = res_df.loc['gt_'+i].sum()-tp
            fp = res_df['pred_'+i].sum()-tp

            cols = res_df.columns[~res_df.columns.isin(['pred_'+i])]
            rows = res_df.index[~res_df.index.isin(['gt_'+i])]
            # print(cols,rows)
            tn = res_df[cols].loc[rows].sum().sum()
            
            tpr = tp/(tp+fn)
            tnr = tn/(tn+fp)
            fpr = fp/(tn+fp)
            fnr = fn/(tp+fn)
            
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1 = 2*((precision*recall)/(precision+recall))
            res_dict["tp"][i] = tp
            res_dict["fp"][i] = fp
            res_dict["fn"][i] = fn
            res_dict["tn"][i] = tn
            res_dict["tpr"][i] = tpr
            res_dict["fpr"][i] = fpr
            res_dict["fnr"][i] = fnr
            res_dict["tnr"][i] = tnr
            res_dict["precision"][i] = precision
            res_dict["recall"][i] = recall
            res_dict["F1"][i] = f1
            res_dict["roc_auc"][i] = roc_auc[i]
        return pd.DataFrame.from_dict(res_dict)

    def initialize_model(self):
        if self.cfg_data["MODEL_ARCH"] == "resnet50":
            from models.ResNet50 import ResNet50_Model as net
            return net(self.cfg_data)
        elif self.cfg_data["MODEL_ARCH"] == "densenet121":
            from models.DenseNet121 import DenseNet121_Model as net
            return net(self.cfg_data)
        return None

    def calculate_roc(self,y_true,y_pred,y_score,class_names,exp_path,save_bool = False):
        classes_bin = [i for i in range(len(class_names))]
        y_true_bin = label_binarize(y_true,classes=classes_bin)
        y_pred_bin = label_binarize(y_pred,classes=classes_bin)
        y_score = np.array(y_score)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[class_names[i]] = metrics.auc(fpr[i], tpr[i])
            
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
        
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(class_names)):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= len(class_names)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
        lw = 2

        # Plot all ROC curves
        if save_bool:
            plt.figure()
            plt.plot(
                fpr["micro"],
                tpr["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
                color="deeppink",
            #     linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr["macro"],
                tpr["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
                color="navy",
            #     linestyle=":",
                linewidth=4,
            )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right",prop={'size': 6})
            # plt.show()
            plt.savefig(exp_path+'/roc.jpg',dpi=300)

            plt.figure()
            colors = cycle(["red","aqua", "darkorange", "cornflowerblue","yellow","lawngreen","hotpink","indigo","lime"])
            for i, color in zip(range(len(class_names)), colors):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(class_names[i], roc_auc[class_names[i]]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right",prop={'size': 6})
            # plt.show()
            plt.savefig(exp_path+'/class_roc.jpg',dpi=300)
        
        return roc_auc
