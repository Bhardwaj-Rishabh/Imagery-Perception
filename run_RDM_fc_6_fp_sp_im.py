import pickle
import scipy.stats as rho
import scipy.io as sio
import numpy as np
import pandas as pd

#--- importing layer data from VGG-16
#pool_3 = pickle.load(open("./layer_data/pool3.p",'rb'))
#pool_4 = pickle.load(open("./layer_data/pool4.p",'rb'))
#pool_5 = pickle.load(open("./layer_data/pool5.p",'rb'))
fc_6 = pickle.load(open("./layer_data/fc_6.p",'rb'))
#fc_7 = pickle.load(open("./layer_data/fc_7.p",'rb'))
#fc_8 = pickle.load(open("./layer_data/fc_8.p",'rb'))

d_fp = dict()
d_sp = dict()
d_im = dict()

p = fc_6

for i in range(101,117):
     d_fp[str(i)] = p[i-101]

for i in range(201,217):
    d_sp[str(i)] = p[i-201]

for i in range(1,17):
    d_im[str(120+i)] = p[i-1]
    d_im[str(220+i)] = p[i-1]

fp = ['111', '107', '113', '105', '103', '102', '112', '110', '113', '113', '113', '110', '102', '115', '106', '115', '115', '109', '115', '111', '109', '106', '109', '104', '110', '107', '112', '116', '111', '108', '115', '104', '113', '103', '103', '105', '110', '113', '102', '109', '115', '112', '104', '105', '108', '116', '116', '116', '106', '103', '112', '116', '110', '101', '106', '105', '108', '103', '105', '106', '106', '102', '107', '113', '112', '115', '102', '105', '113', '110', '112', '101', '107', '103', '108', '105', '111', '103', '113', '102', '107', '108', '110', '111', '101', '110', '104', '104', '111', '107', '110', '108', '111', '116', '109', '105', '107', '106', '115', '114', '101', '107', '102', '113', '110', '114', '103', '101', '106', '108', '111', '105', '106', '105', '106', '107', '114', '114', '115', '116', '110', '112', '112', '104', '106', '115', '116', '116', '110', '113', '103', '112', '114', '106', '103', '102', '114', '116', '101', '112', '104', '111', '108', '105', '116', '115', '102', '101', '104', '111', '114', '114', '112', '115', '107', '112', '102', '116', '109', '116', '101', '114', '104', '110', '109', '109', '105', '104', '108', '113', '107', '103', '102', '114', '107', '116', '101', '114', '102', '113', '105', '112', '108', '114', '110', '109', '108', '101', '113', '102', '114', '102', '115', '105', '112', '104', '101', '107', '114', '108', '108', '113', '101', '104', '106', '101', '104', '101', '105']

sp = ['206', '213', '202', '214', '212', '216', '205', '202', '203', '204', '204', '203', '213', '206', '209', '205', '205', '206', '207', '204', '201', '211', '203', '212', '204', '212', '204', '202', '203', '210', '202', '215', '205', '215', '212', '210', '201', '206', '209', '207', '208', '203', '215', '216', '211', '201', '206', '201', '213', '211', '207', '203', '206', '214', '214', '213', '209', '213', '212', '211', '214', '210', '211', '201', '208', '206', '216', '215', '208', '205', '205', '210', '211', '211', '214', '211', '201', '214', '208', '215', '212', '209', '203', '204', '215', '206', '213', '216', '203', '210', '201', '214', '202', '204', '202', '216', '210', '212', '207', '207', '211', '209', '211', '206', '207', '203', '210', '212', '216', '210', '206', '211', '215', '210', '216', '209', '208', '205', '201', '207', '202', '207', '206', '210', '210', '204', '207', '208', '204', '207', '210', '204', '202', '210', '215', '211', '207', '205', '213', '202', '211', '201', '215', '215', '205', '204', '214', '216', '209', '205', '204', '205', '202', '208', '213', '206', '214', '202', '205', '206', '214', '204', '212', '207', '207', '208', '209', '209', '211', '207', '216', '216', '212', '201', '216', '203', '215', '206', '215', '203', '212', '201', '213', '208', '208', '203', '212', '209', '205', '212', '201', '210', '202', '214', '201', '213', '210', '214', '202', '212', '213', '202', '209', '211', '215', '213', '214', '212', '209']

im = ['226', '127', '222', '234', '232', '236', '132', '222', '133', '224', '224', '130', '122', '226', '126', '135', '135', '226', '135', '224', '129', '126', '129', '232', '224', '232', '224', '222', '131', '230', '222', '124', '133', '123', '232', '230', '130', '226', '122', '129', '228', '132', '124', '236', '128', '136', '226', '136', '126', '123', '132', '136', '226', '234', '234', '125', '128', '123', '232', '126', '234', '230', '127', '133', '228', '226', '236', '125', '228', '130', '132', '230', '127', '123', '234', '125', '131', '234', '228', '122', '232', '128', '130', '224', '121', '226', '124', '236', '131', '230', '130', '234', '222', '224', '222', '236', '230', '232', '135', '134', '121', '127', '122', '226', '130', '134', '230', '232', '236', '230', '226', '125', '126', '230', '236', '127', '228', '134', '135', '136', '222', '132', '226', '230', '230', '224', '136', '228', '224', '133', '230', '224', '222', '230', '123', '122', '134', '136', '121', '222', '124', '131', '128', '125', '136', '224', '234', '236', '124', '131', '224', '134', '222', '228', '127', '226', '234', '222', '129', '226', '234', '224', '232', '130', '129', '228', '125', '124', '128', '133', '236', '236', '232', '134', '236', '136', '121', '226', '122', '133', '232', '132', '128', '228', '228', '129', '232', '121', '133', '232', '134', '230', '222', '234', '132', '124', '230', '234', '222', '232', '128', '222', '121', '124', '126', '121', '234', '232', '125']


mat_fp = []
mat_sp = []
mat_im = []

for i in fp:
     m = []
     for j in fp:
        m.append(1-rho.pearsonr(d_fp[i],d_fp[j])[0])
     mat_fp.append(m)

for i in sp:
    m = []
    for j in sp:
        m.append(1-rho.pearsonr(d_sp[i],d_sp[j])[0])
    mat_sp.append(m)

for i in im:
    m = []
    for j in im:
        m.append(1-rho.pearsonr(d_im[i],d_im[j])[0])
    mat_im.append(m)


fp_sp_im_rdm = [mat_fp,mat_sp,mat_im]
pickle.dump(fp_sp_im_rdm,open('./Data_Python/fp_sp_im_rdm.p','wb'))

el_fp = sio.loadmat("./Data_Matlab/indices_fp.mat")['el2'][0]
el_sp = sio.loadmat("./Data_Matlab/indices_sp.mat")['el3'][0]
el_im = sio.loadmat("./Data_Matlab/indices_im.mat")['el4'][0]

el_fp = [i - 1 for i in el_fp]
el_sp = [i - 1 for i in el_sp]
el_im = [i - 1 for i in el_im]

meg_RDM_fp = sio.loadmat("./Data_Matlab/RDM_datafp.mat")['RDM_data']
meg_RDM_sp = sio.loadmat("./Data_Matlab/RDM_datasp.mat")['RDM_data']
meg_RDM_im = sio.loadmat("./Data_Matlab/RDM_dataim.mat")['RDM_data']

RDM_fp = []
RDM_sp = []
RDM_im = []

time_stamps_fp = meg_RDM_fp.shape[2]  # i.e. 360
time_stamps_sp = meg_RDM_sp.shape[2]
time_stamps_im = meg_RDM_im.shape[2]

for i in range(len(el_fp)):
    for j in range(len(el_fp)):
        for t in range(time_stamps_fp):
            RDM_fp.append(1-rho.pearsonr(meg_RDM_fp[el_fp[i],:,t],meg_RDM_fp[el_fp[j],:,t])[0])



pickle.dump(RDM_fp,open('./Data_Python/RDM_FP_FINAL.p','wb'))



for i in range(len(el_sp)):
    for j in range(len(el_sp)):
        for t in range(time_stamps_sp):
            RDM_sp.append(1-rho.pearsonr(meg_RDM_sp[el_sp[i],:,t],meg_RDM_sp[el_sp[j],:,t])[0])



pickle.dump(RDM_sp,open('./Data_Python/RDM_SP_FINAL.p','wb'))


for i in range(len(el_im)):
    for j in range(len(el_im)):
        for t in range(time_stamps_im):
            RDM_im.append(1-rho.pearsonr(meg_RDM_im[el_im[i],:,t],meg_RDM_im[el_im[j],:,t])[0])



pickle.dump(RDM_im,open('./Data_Python/RDM_IM_FINAL.p','wb'))


corr_time_fp_fp = np.zeros(time_stamps_fp)
corr_time_sp_fp = np.zeros(time_stamps_fp)
corr_time_sp_sp = np.zeros(time_stamps_sp)
corr_time_im_fp = np.zeros(time_stamps_im)
corr_time_im_sp = np.zeros(time_stamps_im)
corr_time_im_im = np.zeros(time_stamps_im)

RDM_fp = np.reshape(RDM_fp,(209,209,time_stamps_fp))
RDM_sp = np.reshape(RDM_sp,(209,209,time_stamps_sp))
RDM_im = np.reshape(RDM_im,(209,209,time_stamps_im))

m_fp = np.array(mat_fp)
m_sp = np.array(mat_sp)
m_im = np.array(mat_im)

for t in range(time_stamps_fp):
     print(t)
     m_reshaped = np.reshape(m_fp,(209*209,))
     meg_t = np.reshape(RDM_fp[:,:,t],(209*209,))
     corr_time_fp_fp[t] = rho.spearmanr((m_reshaped),(meg_t))[0]

for t in range(time_stamps_sp):
    print(t)
    m_reshaped = np.reshape(m_fp,(209*209,))
    meg_t = np.reshape(RDM_sp[:,:,t],(209*209,))
    corr_time_sp_fp[t] = rho.spearmanr((m_reshaped),(meg_t))[0]
    m_reshaped = np.reshape(m_sp,(209*209,))
    corr_time_sp_sp[t] = rho.spearmanr((m_reshaped),(meg_t))[0]

for t in range(time_stamps_im):
    print(t)
    m_reshaped = np.reshape(m_fp,(209*209,))
    meg_t = np.reshape(RDM_im[:,:,t],(209*209,))
    corr_time_im_fp[t] = rho.spearmanr((m_reshaped),(meg_t))[0]
    m_reshaped = np.reshape(m_sp,(209*209,))
    corr_time_im_sp[t] = rho.spearmanr((m_reshaped),(meg_t))[0]
    m_reshaped = np.reshape(m_im,(209*209,))
    corr_time_im_im[t] = rho.spearmanr((m_reshaped),(meg_t))[0]

pickle.dump(corr_time_fp_fp, open("./Results_fc_6_fp_sp_im/corr_time_fp_fp.p",'wb'))
pickle.dump(corr_time_sp_fp, open("./Results_fc_6_fp_sp_im/corr_time_sp_fp.p",'wb'))
pickle.dump(corr_time_sp_sp, open("./Results_fc_6_fp_sp_im/corr_time_sp_sp.p",'wb'))
pickle.dump(corr_time_im_fp, open("./Results_fc_6_fp_sp_im/corr_time_im_fp.p",'wb'))
pickle.dump(corr_time_im_sp, open("./Results_fc_6_fp_sp_im/corr_time_im_sp.p",'wb'))
pickle.dump(corr_time_im_im, open("./Results_fc_6_fp_sp_im/corr_time_im_im.p",'wb'))

pd.DataFrame(corr_time_fp_fp).to_csv('./Results_fc_6_fp_sp_im/corr_time_fp_fp.csv')
pd.DataFrame(corr_time_sp_fp).to_csv('./Results_fc_6_fp_sp_im/corr_time_sp_fp.csv')
pd.DataFrame(corr_time_sp_sp).to_csv('./Results_fc_6_fp_sp_im/corr_time_sp_sp.csv')
pd.DataFrame(corr_time_im_fp).to_csv('./Results_fc_6_fp_sp_im/corr_time_im_fp.csv')
pd.DataFrame(corr_time_im_sp).to_csv('./Results_fc_6_fp_sp_im/corr_time_im_sp.csv')
pd.DataFrame(corr_time_im_im).to_csv('./Results_fc_6_fp_sp_im/corr_time_im_im.csv')






