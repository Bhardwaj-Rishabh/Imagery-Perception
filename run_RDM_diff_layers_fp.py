import pickle
import scipy.stats as rho
import scipy.io as sio
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

pool_3 = pickle.load(open("./Results_all_layers_fp_fp/pool_3.p",'rb'))
pool_4 = pickle.load(open("./Results_all_layers_fp_fp/pool_4.p",'rb'))
pool_5 = pickle.load(open("./Results_all_layers_fp_fp/pool_5.p",'rb'))
fc_6 = pickle.load(open("./Results_all_layers_fp_fp/fc_6.p",'rb'))
fc_7 = pickle.load(open("./Results_all_layers_fp_fp/fc_7.p",'rb'))
fc_8 = pickle.load(open("./Results_all_layers_fp_fp/fc_8.p",'rb'))

def corr_fp_fp_diff_layers(layer,p):
    d_fp = dict()

    for i in range(101,117):
         d_fp[str(i)] = p[i-101]

    fp = ['111', '107', '113', '105', '103', '102', '112', '110', '113', '113', '113', '110', '102', '115', '106', '115', '115', '109', '115', '111', '109', '106', '109', '104', '110', '107', '112', '116', '111', '108', '115', '104', '113', '103', '103', '105', '110', '113', '102', '109', '115', '112', '104', '105', '108', '116', '116', '116', '106', '103', '112', '116', '110', '101', '106', '105', '108', '103', '105', '106', '106', '102', '107', '113', '112', '115', '102', '105', '113', '110', '112', '101', '107', '103', '108', '105', '111', '103', '113', '102', '107', '108', '110', '111', '101', '110', '104', '104', '111', '107', '110', '108', '111', '116', '109', '105', '107', '106', '115', '114', '101', '107', '102', '113', '110', '114', '103', '101', '106', '108', '111', '105', '106', '105', '106', '107', '114', '114', '115', '116', '110', '112', '112', '104', '106', '115', '116', '116', '110', '113', '103', '112', '114', '106', '103', '102', '114', '116', '101', '112', '104', '111', '108', '105', '116', '115', '102', '101', '104', '111', '114', '114', '112', '115', '107', '112', '102', '116', '109', '116', '101', '114', '104', '110', '109', '109', '105', '104', '108', '113', '107', '103', '102', '114', '107', '116', '101', '114', '102', '113', '105', '112', '108', '114', '110', '109', '108', '101', '113', '102', '114', '102', '115', '105', '112', '104', '101', '107', '114', '108', '108', '113', '101', '104', '106', '101', '104', '101', '105']

    mat_fp = []

    for i in fp:
         m = []
         for j in fp:
            m.append(1-rho.pearsonr(d_fp[i],d_fp[j])[0])
         mat_fp.append(m)

    el_fp = sio.loadmat("./Data_Matlab/indices_fp.mat")['el2'][0]
    el_fp = [i - 1 for i in el_fp]

    meg_RDM_fp = sio.loadmat("./Data_Matlab/RDM_datafp.mat")['RDM_data']
    RDM_fp = []

    time_stamps_fp = meg_RDM_fp.shape[2]  # i.e. 360

    RDM_fp = pickle.load(open('./Data_Python/RDM_FP_FINAL.p','rb'))

    corr_time_fp_fp = np.zeros(time_stamps_fp)

    RDM_fp = np.reshape(RDM_fp,(209,209,time_stamps_fp))

    m_fp = np.array(mat_fp)

    for t in range(time_stamps_fp):
         print(t)
         m_reshaped = np.reshape(m_fp,(209*209,))
         meg_t = np.reshape(RDM_fp[:,:,t],(209*209,))
         corr_time_fp_fp[t] = rho.spearmanr((m_reshaped),(meg_t))[0]

    pickle.dump(corr_time_fp_fp, open("./Results_all_layers_fp_fp/"+layer+"corr_time_fp_fp.p",'wb'))
    pd.DataFrame(corr_time_fp_fp).to_csv("./Results_all_layers_fp_fp/"+layer+"corr_time_fp_fp.csv")


corr_fp_fp_diff_layers('pool_3',pool_3)
corr_fp_fp_diff_layers('pool_3',pool_4)
corr_fp_fp_diff_layers('pool_3',pool_5)
corr_fp_fp_diff_layers('fc_6.p',fc_6)
corr_fp_fp_diff_layers('fc_7.p',fc_7)
corr_fp_fp_diff_layers('fc_8.p',fc_8)




