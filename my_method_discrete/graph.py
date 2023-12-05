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
default_cycler = (cycler(color=['#ff7f0e', '#2ca02c','#f3533a', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','#f3533a', '#fa9f42', '#8ad879', '#5acfc9']) )  
plt.rc('axes', prop_cycle=default_cycler)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iteration_number=(np.array(list(range(5))))


data_no_2 = np.loadtxt("./results/result_nohole_d2.csv", delimiter=',')
data_no_2_mean=np.sum(data_no_2,axis=0)/data_no_2.shape[0]
data_no_2_sd=np.sqrt(np.var(data_no_2,axis=0))*5.0

data_no_1 = np.loadtxt("./results/result_nohole_d1.csv", delimiter=',')
data_no_1_mean=np.sum(data_no_1,axis=0)/data_no_1.shape[0]
data_no_1_sd=np.sqrt(np.var(data_no_1,axis=0))*5.0

data_hole_2 = np.loadtxt("./results/result_hole_d2.csv", delimiter=',')
data_hole_2_mean=np.sum(data_hole_2,axis=0)/data_hole_2.shape[0]
data_hole_2_sd=np.sqrt(np.var(data_hole_2,axis=0))*5.0

data_hole_1 = np.loadtxt("./results/result_hole_d1.csv", delimiter=',')
data_hole_1_mean=np.sum(data_hole_1,axis=0)/data_hole_1.shape[0]
data_hole_1_sd=np.sqrt(np.var(data_hole_1,axis=0))*5.0

data_no_2_from0 = np.loadtxt("./results/result_nohole_d2_from0.csv", delimiter=',')
data_no_2_from0_mean=np.sum(data_no_2_from0,axis=0)/data_no_2_from0.shape[0]
data_no_2_from0_sd=np.sqrt(np.var(data_no_2_from0,axis=0))*5.0

data_no_1_from0 = np.loadtxt("./results/result_nohole_d1_from0.csv", delimiter=',')
data_no_1_from0_mean=np.sum(data_no_1_from0,axis=0)/data_no_1_from0.shape[0]
data_no_1_from0_sd=np.sqrt(np.var(data_no_1_from0,axis=0))*5.0

data_hole_2_from0 = np.loadtxt("./results/result_hole_d2_from0.csv", delimiter=',')
data_hole_2_from0_mean=np.sum(data_hole_2_from0,axis=0)/data_hole_2_from0.shape[0]
data_hole_2_from0_sd=np.sqrt(np.var(data_hole_2_from0,axis=0))*5.0

data_hole_1_from0 = np.loadtxt("./results/result_hole_d1_from0.csv", delimiter=',')
data_hole_1_from0_mean=np.sum(data_hole_1_from0,axis=0)/data_hole_1_from0.shape[0]
data_hole_1_from0_sd=np.sqrt(np.var(data_hole_1_from0,axis=0))*5.0

data_no_2_optimal = np.loadtxt("./results/result_nohole_d2_optimal.csv", delimiter=',')
data_no_1_optimal= np.loadtxt("./results/result_nohole_d1_optimal.csv", delimiter=',')
data_hole_2_optimal = np.loadtxt("./results/result_hole_d2_optimal.csv", delimiter=',')
data_hole_1_optimal = np.loadtxt("./results/result_hole_d1_optimal.csv", delimiter=',')



plt.rcParams.update({'font.size': 24})
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus']=False 

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_hole_1_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL (ours)")
ax.fill_between(axis,data_hole_1_mean-data_hole_1_sd,data_hole_1_mean+data_hole_1_sd,alpha=0.2)
plt.plot(axis,data_hole_1_from0_mean,'-.',marker="x", linewidth=2.5,markersize=8,label="Starting from scratch")
ax.fill_between(axis,data_hole_1_from0_mean-data_hole_1_from0_sd,data_hole_1_from0_mean+data_hole_1_from0_sd,alpha=0.2)
plt.plot(axis,data_hole_1_optimal,'--', linewidth=2.5 ,label="Optimal task-specific policies")
#plt.plot(axis,boil_clean,linestyle=(0,(3, 1, 1, 1, 1, 1)),label="BOIL with MOML")

#plt.xticks(np.arange(0,iterations,40))
plt.title('Large task variance ($\mathcal{A l g}^{(1)}$ applied)',size=28)
plt.xlabel('Number of adaptaion steps',size=28)
plt.ylabel("Accumulated reward",size=28)
plt.ylim(-0.5,1.75)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.129, right=0.993, top=0.924, bottom=0.129)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
plt.savefig('hole_1.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_no_1_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL (ours)")
ax.fill_between(axis,data_no_1_mean-data_no_1_sd,data_no_1_mean+data_no_1_sd,alpha=0.2)
plt.plot(axis,data_no_1_from0_mean,'-.',marker="x", linewidth=2.5,markersize=8,label="Starting from scratch")
ax.fill_between(axis,data_no_1_from0_mean-data_no_1_from0_sd,data_no_1_from0_mean+data_no_1_from0_sd,alpha=0.2)
plt.plot(axis,data_no_1_optimal,'--', linewidth=2.5 ,label="Optimal task-specific policies")
#plt.plot(axis,boil_clean,linestyle=(0,(3, 1, 1, 1, 1, 1)),label="BOIL with MOML")

#plt.xticks(np.arange(0,iterations,40))
plt.title('Small task variance ($\mathcal{A l g}^{(1)}$ applied)',size=28)
plt.xlabel('Number of adaptaion steps',size=28)
plt.ylabel("Accumulated reward",size=28)
plt.ylim(-0.25,1.75)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.129, right=0.993, top=0.924, bottom=0.129)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
plt.savefig('nohole_1.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_hole_2_mean,'-',marker="o",markersize=8, linewidth=2.5,label="BO-MRL (ours)")
ax.fill_between(axis,data_hole_2_mean-data_hole_2_sd,data_hole_2_mean+data_hole_2_sd,alpha=0.2)
plt.plot(axis,data_hole_2_from0_mean,'-.',marker="x",markersize=8, linewidth=2.5,label="Starting from scratch")
ax.fill_between(axis,data_hole_2_from0_mean-data_hole_2_from0_sd,data_hole_2_from0_mean+data_hole_2_from0_sd,alpha=0.2)
plt.plot(axis,data_hole_2_optimal,'--' , linewidth=2.5 ,label="Optimal task-specific policies")
#plt.plot(axis,boil_clean,linestyle=(0,(3, 1, 1, 1, 1, 1)),label="BOIL with MOML")

#plt.xticks(np.arange(0,iterations,40))
plt.title('Large task variance ($\mathcal{A l g}^{(2)}$ applied)',size=28)
plt.xlabel('Number of adaptaion steps',size=28)
plt.ylabel("Accumulated reward",size=28)
plt.ylim(-0.5,1.75)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.129, right=0.993, top=0.924, bottom=0.129)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
plt.savefig('hole_2.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_no_2_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL (ours)")
ax.fill_between(axis,data_no_2_mean-data_no_2_sd,data_no_2_mean+data_no_2_sd,alpha=0.2)
plt.plot(axis,data_no_2_from0_mean,'-.',marker="x", linewidth=2.5,markersize=8,label="Starting from scratch")
ax.fill_between(axis,data_no_2_from0_mean-data_no_2_from0_sd,data_no_2_from0_mean+data_no_2_from0_sd,alpha=0.2)
plt.plot(axis,data_no_2_optimal,'--', linewidth=2.5 ,label="Optimal task-specific policies")
#plt.plot(axis,boil_clean,linestyle=(0,(3, 1, 1, 1, 1, 1)),label="BOIL with MOML")

#plt.xticks(np.arange(0,iterations,40))
plt.title('Small task variance ($\mathcal{A l g}^{(2)}$ applied)',size=28)
plt.xlabel('Number of adaptaion steps',size=28)
plt.ylabel("Accumulated reward",size=28)
plt.ylim(-0.25,1.75)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.129, right=0.993, top=0.924, bottom=0.129)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
plt.savefig('nohole_2.pdf') 
plt.show()


axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_no_2_mean,'-',label="CML (ours)")
plt.plot(axis,data_no_2_from0_mean,'-.',label="Protonet with MOML")
plt.plot(axis,data_no_2_optimal,'--',label="MAML with MOML")
#plt.plot(axis,boil_b,linestyle=(0,(3, 1, 1, 1, 1, 1)),label="BOIL with MOML")

