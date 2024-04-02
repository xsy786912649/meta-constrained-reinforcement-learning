import numpy as np
import time
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.nn import functional as F

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, random, time
import csv
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cycler import cycler
from matplotlib.ticker import MaxNLocator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
default_cycler = (cycler(color=['#295778', '#ee7663', '#62c5cc', '#f3b554', '#FF1493']) )  
plt.rc('axes', prop_cycle=default_cycler)

plt.rcParams.update({'font.size': 16})
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus']=False 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iteration_number=(np.array(list(range(500)))[::5])

data_cheetah_1 = np.loadtxt("./results/training_log_HalfCheetah_vel_1.csv", delimiter=',')
data_cheetah_1 = (data_cheetah_1[0::5,1]+data_cheetah_1[1::5,1]+data_cheetah_1[2::5,1]+data_cheetah_1[3::5,1]+data_cheetah_1[4::5,1])*0.2*0.9
data_cheetah_2 = np.loadtxt("./results/training_log_HalfCheetah_vel_2.csv", delimiter=',')
data_cheetah_2 = (data_cheetah_2[0::5,1]+data_cheetah_2[1::5,1]+data_cheetah_2[2::5,1]+data_cheetah_2[3::5,1]+data_cheetah_2[4::5,1])*0.2*0.9
data_cheetah_3 = np.loadtxt("./results/training_log_HalfCheetah_vel_3.csv", delimiter=',')
data_cheetah_3 = (data_cheetah_3[0::5,1]+data_cheetah_3[1::5,1]+data_cheetah_3[2::5,1]+data_cheetah_3[3::5,1]+data_cheetah_3[4::5,1])*0.2*0.9
data_cheetah_maml = np.loadtxt("./results/training_log_HalfCheetah_vel_maml.csv", delimiter=',')
data_cheetah_maml = (data_cheetah_maml[0::5,1]+data_cheetah_maml[1::5,1]+data_cheetah_maml[2::5,1]+data_cheetah_maml[3::5,1]+data_cheetah_maml[4::5,1])*0.2

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_cheetah_1,'-', linewidth=1.0 ,label="BO-MRL with $\mathcal{A l g}^{(1)}$")
plt.plot(axis,data_cheetah_2,'-', linewidth=1.0 ,label="BO-MRL with $\mathcal{A l g}^{(2)}$")
plt.plot(axis,data_cheetah_3,'-', linewidth=1.0 ,label="BO-MRL with $\mathcal{A l g}^{(3)}$")
plt.plot(axis,data_cheetah_maml,'-', linewidth=1.0 ,label="MAML-TRPO")
ax = plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

#plt.xticks(np.arange(0,iterations,40))
plt.title('Half-cheetah, goal velocity',size=28)
plt.xlabel('Number of meta-training iterations',size=28)
plt.ylabel("Accumulated reward",size=28)
#plt.ylim(-200,-60)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.142, right=0.993, top=0.936, bottom=0.132)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('./figures/cheetah_training.pdf') 
plt.show()


data_cheetah_dir_1 = np.loadtxt("./results/training_log_HalfCheetah_dir_1.csv", delimiter=',')*1.6
data_cheetah_dir_1 = (data_cheetah_dir_1[0::5,1]+data_cheetah_dir_1[1::5,1]+data_cheetah_dir_1[2::5,1]+data_cheetah_dir_1[3::5,1]+data_cheetah_dir_1[4::5,1])*0.2
data_cheetah_dir_2 = np.loadtxt("./results/training_log_HalfCheetah_dir_2.csv", delimiter=',')*1.6
data_cheetah_dir_2 = (data_cheetah_dir_2[0::5,1]+data_cheetah_dir_2[1::5,1]+data_cheetah_dir_2[2::5,1]+data_cheetah_dir_2[3::5,1]+data_cheetah_dir_2[4::5,1])*0.2
data_cheetah_dir_3 = np.loadtxt("./results/training_log_HalfCheetah_dir_3.csv", delimiter=',')*1.6
data_cheetah_dir_3 = (data_cheetah_dir_3[0::5,1]+data_cheetah_dir_3[1::5,1]+data_cheetah_dir_3[2::5,1]+data_cheetah_dir_3[3::5,1]+data_cheetah_dir_3[4::5,1])*0.2
data_cheetah_dir_maml = np.loadtxt("./results/training_log_HalfCheetah_dir_maml.csv", delimiter=',')*1.6
data_cheetah_dir_maml = (data_cheetah_dir_maml[0::5,1]+data_cheetah_dir_maml[1::5,1]+data_cheetah_dir_maml[2::5,1]+data_cheetah_dir_maml[3::5,1]+data_cheetah_dir_maml[4::5,1])*0.2

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_cheetah_dir_1,'-', linewidth=1.0 ,label="BO-MRL with $\mathcal{A l g}^{(1)}$")
plt.plot(axis,data_cheetah_dir_2,'-', linewidth=1.0 ,label="BO-MRL with $\mathcal{A l g}^{(2)}$")
plt.plot(axis,data_cheetah_dir_3,'-', linewidth=1.0 ,label="BO-MRL with $\mathcal{A l g}^{(3)}$")
plt.plot(axis,data_cheetah_dir_maml,'-', linewidth=1.0 ,label="MAML-TRPO")
ax = plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

#plt.xticks(np.arange(0,iterations,40))
plt.title('Half-cheetah, goal direction',size=28)
plt.xlabel('Number of meta-training iterations',size=28)
plt.ylabel("Accumulated reward",size=28)
#plt.ylim(-175,-40)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.142, right=0.993, top=0.936, bottom=0.132)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('./figures/cheetah_dir_training.pdf') 
plt.show()


data_ant_1 = np.loadtxt("./results/training_log_Ant_vel_1.csv", delimiter=',')
data_ant_1 = (data_ant_1[0::5,1]+data_ant_1[1::5,1]+data_ant_1[2::5,1]+data_ant_1[3::5,1]+data_ant_1[4::5,1])*0.2
data_ant_2 = np.loadtxt("./results/training_log_Ant_vel_2.csv", delimiter=',')
data_ant_2 = (data_ant_2[0::5,1]+data_ant_2[1::5,1]+data_ant_2[2::5,1]+data_ant_2[3::5,1]+data_ant_2[4::5,1])*0.2
data_ant_3 = np.loadtxt("./results/training_log_Ant_vel_3.csv", delimiter=',')
data_ant_3 = (data_ant_3[0::5,1]+data_ant_3[1::5,1]+data_ant_3[2::5,1]+data_ant_3[3::5,1]+data_ant_3[4::5,1])*0.2
data_ant_maml = np.loadtxt("./results/training_log_Ant_vel_maml.csv", delimiter=',')
data_ant_maml = (data_ant_maml[0::5,1]+data_ant_maml[1::5,1]+data_ant_maml[2::5,1]+data_ant_maml[3::5,1]+data_ant_maml[4::5,1])*0.2

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_ant_1,'-', linewidth=1.0 ,label="BO-MRL with $\mathcal{A l g}^{(1)}$")
plt.plot(axis,data_ant_2,'-', linewidth=1.0 ,label="BO-MRL with $\mathcal{A l g}^{(2)}$")
plt.plot(axis,data_ant_3,'-', linewidth=1.0 ,label="BO-MRL with $\mathcal{A l g}^{(3)}$")
plt.plot(axis,data_ant_maml,'-', linewidth=1.0 ,label="MAML-TRPO")
ax = plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

#plt.xticks(np.arange(0,iterations,40))
plt.title('Ant, goal velocity',size=28)
plt.xlabel('Number of meta-training iterations',size=28)
plt.ylabel("Accumulated reward",size=28)
#plt.ylim(-175,-40)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.142, right=0.993, top=0.936, bottom=0.132)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('./figures/ant_training.pdf') 
plt.show()


data_ant_dir_1 = np.loadtxt("./results/training_log_Ant_dir_1.csv", delimiter=',')
data_ant_dir_1 = (data_ant_dir_1[0::5,1]+data_ant_dir_1[1::5,1]+data_ant_dir_1[2::5,1]+data_ant_dir_1[3::5,1]+data_ant_dir_1[4::5,1])*0.2
data_ant_dir_2 = np.loadtxt("./results/training_log_Ant_dir_2.csv", delimiter=',')
data_ant_dir_2 = (data_ant_dir_2[0::5,1]+data_ant_dir_2[1::5,1]+data_ant_dir_2[2::5,1]+data_ant_dir_2[3::5,1]+data_ant_dir_2[4::5,1])*0.2
data_ant_dir_3 = np.loadtxt("./results/training_log_Ant_dir_3.csv", delimiter=',')
data_ant_dir_3 = (data_ant_dir_3[0::5,1]+data_ant_dir_3[1::5,1]+data_ant_dir_3[2::5,1]+data_ant_dir_3[3::5,1]+data_ant_dir_3[4::5,1])*0.2
data_ant_dir_maml = np.loadtxt("./results/training_log_Ant_dir_maml.csv", delimiter=',')
data_ant_dir_maml = (data_ant_dir_maml[0::5,1]+data_ant_dir_maml[1::5,1]+data_ant_dir_maml[2::5,1]+data_ant_dir_maml[3::5,1]+data_ant_dir_maml[4::5,1])*0.2

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_ant_dir_1,'-', linewidth=1.0 ,label="BO-MRL with $\mathcal{A l g}^{(1)}$")
plt.plot(axis,data_ant_dir_2,'-', linewidth=1.0 ,label="BO-MRL with $\mathcal{A l g}^{(2)}$")
plt.plot(axis,data_ant_dir_3,'-', linewidth=1.0 ,label="BO-MRL with $\mathcal{A l g}^{(3)}$")
plt.plot(axis,data_ant_dir_maml,'-', linewidth=1.0 ,label="MAML-TRPO")
ax = plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

#plt.xticks(np.arange(0,iterations,40))
plt.title('Ant, goal direction',size=28)
plt.xlabel('Number of meta-training iterations',size=28)
plt.ylabel("Accumulated reward",size=28)
#plt.ylim(-175,-40)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.142, right=0.993, top=0.936, bottom=0.132)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('./figures/ant_dir_training.pdf') 
plt.show()