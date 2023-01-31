import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score


path_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
path_dataoriginal = os.path.join(path_root, 'data_original')
path_dataruchir = os.path.join(path_dataoriginal, 'data_ruchir')

with open(os.path.join(path_dataruchir, 'AllTrTesData10fold.pkl'), 'rb') as f:
    package = pickle.load(f)


lst_test_rocauc = list()
lst_val_rocauc = list()
lst_models = list()
lst_models.append(('s',MinMaxScaler()))
lst_models.append(('fs',SelectKBest(score_func=chi2,k=32)))
lst_models.append(('guassian',StandardScaler()))
lst_models.append(('m',LogisticRegression(multi_class='multinomial')))

pip=Pipeline(lst_models)

for c_kf in range(10):

    df_x_train_kf, df_x_test_kf = pd.DataFrame(package[0][c_kf][0]), pd.DataFrame(package[1][c_kf][0])
    df_y_train_kf, df_y_test_kf = pd.DataFrame(package[2][c_kf][0] -1), pd.DataFrame(package[3][c_kf][0] -1)
    # counts = df_x_train_kf.nunique()
    # # record columns to delete
    # to_del = [i for i,v in enumerate(counts) if (float(v)/df_x_train_kf.shape[0]*100) < 0.05]
 
    # # drop useless columns
    # df_x_train_kf.drop(to_del, axis=1, inplace=True)
    # df_x_test_kf.drop(to_del, axis=1, inplace=True)
    pip.fit(df_x_train_kf, df_y_train_kf.values.ravel())
   
    y_predproba = pip.predict_proba(df_x_test_kf)
    
    lst_val_rocauc.append(roc_auc_score(df_y_test_kf.values.ravel(),y_predproba,multi_class='ovr'))

print(round(np.average(lst_val_rocauc),2))
