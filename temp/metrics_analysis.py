import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

metrics_test_path = './metrics/test_metrics_';
metrics_train_path = './metrics/train_metrics_';
metrics_ABGR_path = './metrics/ABGRtest_metrics';

dftst = pd.DataFrame();
dftrn = pd.DataFrame();

for i in range(21):
    df = pd.read_csv(metrics_test_path + str(i) + '.csv',names = ['id',str(i)]);
    dftst[str(i)] = df[str(i)];
    #df = pd.read_csv(metrics_train_path + str(i) + '.csv',names = ['id',str(i)]);
    #dftrn[str(i)] = df[str(i)];


dfabgr = pd.read_csv(metrics_ABGR_path + '.csv',names = ['id','iou']);