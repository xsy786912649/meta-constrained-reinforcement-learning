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
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
default_cycler = (cycler(color=['#295778', '#ee7663', '#62c5cc', '#f3b554', '#cc87f8']) )  
plt.rc('axes', prop_cycle=default_cycler)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iteration_number=(np.array(list(range(5))))

data_cheetah_2 = np.loadtxt("./results/result_cheetah_2.csv", delimiter=',')
data_cheetah_2_mean=np.sum(data_cheetah_2,axis=0)/data_cheetah_2.shape[0]
data_cheetah_2_sd=np.sqrt(np.var(data_cheetah_2,axis=0))

data_cheetah_1 = np.loadtxt("./results/result_cheetah_1.csv", delimiter=',')
data_cheetah_1_mean=np.sum(data_cheetah_1,axis=0)/data_cheetah_1.shape[0]
data_cheetah_1_sd=np.sqrt(np.var(data_cheetah_1,axis=0))

data_cheetah_dir_2 = np.loadtxt("./results/result_cheetah_dir_2.csv", delimiter=',')
data_cheetah_dir_2_mean=np.sum(data_cheetah_dir_2,axis=0)/data_cheetah_dir_2.shape[0]
data_cheetah_dir_2_sd=np.sqrt(np.var(data_cheetah_dir_2,axis=0))

data_cheetah_dir_1 = np.loadtxt("./results/result_cheetah_dir_1.csv", delimiter=',') 
data_cheetah_dir_1_mean=np.sum(data_cheetah_dir_1,axis=0)/data_cheetah_dir_1.shape[0]
data_cheetah_dir_1_sd=np.sqrt(np.var(data_cheetah_dir_1,axis=0))


data_ant_2 = np.loadtxt("./results/result_ant_2.csv", delimiter=',')
data_ant_2_mean=np.sum(data_ant_2,axis=0)/data_ant_2.shape[0]
data_ant_2_sd=np.sqrt(np.var(data_ant_2,axis=0))

data_ant_1 = np.loadtxt("./results/result_ant_1.csv", delimiter=',')
data_ant_1_mean=np.sum(data_ant_1,axis=0)/data_ant_1.shape[0]
data_ant_1_sd=np.sqrt(np.var(data_ant_1,axis=0))

data_ant_dir_2 = np.loadtxt("./results/result_ant_dir_2.csv", delimiter=',')
data_ant_dir_2_mean=np.sum(data_ant_dir_2,axis=0)/data_ant_dir_2.shape[0]
data_ant_dir_2_sd=np.sqrt(np.var(data_ant_dir_2,axis=0))

data_ant_dir_1 = np.loadtxt("./results/result_ant_dir_1.csv", delimiter=',') 
data_ant_dir_1_mean=np.sum(data_ant_dir_1,axis=0)/data_ant_dir_1.shape[0]
data_ant_dir_1_sd=np.sqrt(np.var(data_ant_dir_1,axis=0))


data_cheetah_maml = np.loadtxt("./results/result_cheetah_maml.csv", delimiter=',')
data_cheetah_dir_maml= np.loadtxt("./results/result_cheetah_dir_maml.csv", delimiter=',')
data_ant_maml = np.loadtxt("./results/result_ant_maml.csv", delimiter=',')
data_ant_dir_maml = np.loadtxt("./results/result_ant_dir_maml.csv", delimiter=',')

data_cheetah_emaml = np.loadtxt("./results/result_cheetah_emaml.csv", delimiter=',')
data_cheetah_dir_emaml= np.loadtxt("./results/result_cheetah_dir_emaml.csv", delimiter=',')
data_ant_emaml = np.loadtxt("./results/result_ant_emaml.csv", delimiter=',')
data_ant_dir_emaml = np.loadtxt("./results/result_ant_dir_emaml.csv", delimiter=',')

data_cheetah_ProMP = np.loadtxt("./results/result_cheetah_ProMP.csv", delimiter=',')
data_cheetah_dir_ProMP= np.loadtxt("./results/result_cheetah_dir_ProMP.csv", delimiter=',')
data_ant_ProMP = np.loadtxt("./results/result_ant_ProMP.csv", delimiter=',')
data_ant_dir_ProMP = np.loadtxt("./results/result_ant_dir_ProMP.csv", delimiter=',')

plt.rcParams.update({'font.size': 24})
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus']=False 

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_cheetah_1_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL with $\mathcal{A l g}^{(1)}$")
ax.fill_between(axis,data_cheetah_1_mean-data_cheetah_1_sd,data_cheetah_1_mean+data_cheetah_1_sd,alpha=0.4)
plt.plot(axis,data_cheetah_2_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL with $\mathcal{A l g}^{(2)}$")
ax.fill_between(axis,data_cheetah_2_mean-data_cheetah_2_sd,data_cheetah_2_mean+data_cheetah_2_sd,alpha=0.4)
plt.plot(axis,data_cheetah_emaml,'-.',marker="x", linewidth=2.5,markersize=8,label="E-MAML")
plt.plot(axis,data_cheetah_maml,'-',marker="1",markersize=8, linewidth=2.5 ,label="MAML")
plt.plot(axis,data_cheetah_ProMP,'--', linewidth=2.5 ,label="ProMP")

#plt.xticks(np.arange(0,iterations,40))
plt.title('Half-cheetah, goal velocity',size=28)
plt.xlabel('Number of policy adaptation steps',size=28)
plt.ylabel("Accumulated reward",size=28)
plt.ylim(-150,-50)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.129, right=0.993, top=0.923, bottom=0.126)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('./figures/cheetah.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_cheetah_dir_1_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL with $\mathcal{A l g}^{(1)}$")
ax.fill_between(axis,data_cheetah_dir_1_mean-data_cheetah_dir_1_sd,data_cheetah_dir_1_mean+data_cheetah_dir_1_sd,alpha=0.4)
plt.plot(axis,data_cheetah_dir_2_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL with $\mathcal{A l g}^{(2)}$")
ax.fill_between(axis,data_cheetah_dir_2_mean-data_cheetah_dir_2_sd,data_cheetah_dir_2_mean+data_cheetah_dir_2_sd,alpha=0.4)
plt.plot(axis,data_cheetah_dir_emaml,'-.',marker="x", linewidth=2.5,markersize=8,label="E-MAML")
plt.plot(axis,data_cheetah_dir_maml,'-',marker="1",markersize=8, linewidth=2.5 ,label="MAML")
plt.plot(axis,data_cheetah_dir_ProMP,'--', linewidth=2.5 ,label="ProMP")

#plt.xticks(np.arange(0,iterations,40))
plt.title('Half-cheetah, moving direction',size=28)
plt.xlabel('Number of policy adaptation steps',size=28)
plt.ylabel("Accumulated reward",size=28)
plt.ylim(-150,-50)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.129, right=0.993, top=0.923, bottom=0.126)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('./figures/cheetah_dir.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_ant_1_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL with $\mathcal{A l g}^{(1)}$")
ax.fill_between(axis,data_ant_1_mean-data_ant_1_sd,data_ant_1_mean+data_ant_1_sd,alpha=0.4)
plt.plot(axis,data_ant_2_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL with $\mathcal{A l g}^{(2)}$")
ax.fill_between(axis,data_ant_2_mean-data_ant_2_sd,data_ant_2_mean+data_ant_2_sd,alpha=0.4)
plt.plot(axis,data_ant_emaml,'-.',marker="x", linewidth=2.5,markersize=8,label="E-MAML")
plt.plot(axis,data_ant_maml,'-',marker="1",markersize=8, linewidth=2.5 ,label="MAML")
plt.plot(axis,data_ant_ProMP,'--', linewidth=2.5 ,label="ProMP")

#plt.xticks(np.arange(0,iterations,40))
plt.title('Ant, goal velocity',size=28)
plt.xlabel('Number of policy adaptation steps',size=28)
plt.ylabel("Accumulated reward",size=28)
plt.ylim(-150,-50)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.129, right=0.993, top=0.923, bottom=0.126)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('./figures/ant.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_ant_dir_1_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL with $\mathcal{A l g}^{(1)}$")
ax.fill_between(axis,data_ant_dir_1_mean-data_ant_dir_1_sd,data_ant_dir_1_mean+data_ant_dir_1_sd,alpha=0.4)
plt.plot(axis,data_ant_dir_2_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL with $\mathcal{A l g}^{(2)}$")
ax.fill_between(axis,data_ant_dir_2_mean-data_ant_dir_2_sd,data_ant_dir_2_mean+data_ant_dir_2_sd,alpha=0.4)
plt.plot(axis,data_ant_dir_emaml,'-.',marker="x", linewidth=2.5,markersize=8,label="E-MAML")
plt.plot(axis,data_ant_dir_maml,'-',marker="1",markersize=8, linewidth=2.5 ,label="MAML")
plt.plot(axis,data_ant_dir_ProMP,'--', linewidth=2.5 ,label="ProMP")

#plt.xticks(np.arange(0,iterations,40))
plt.title('Ant, moving direction',size=28)
plt.xlabel('Number of policy adaptation steps',size=28)
plt.ylabel("Accumulated reward",size=28)
plt.ylim(-150,-50)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.129, right=0.993, top=0.923, bottom=0.126)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('./figures/ant_dir.pdf') 
plt.show()