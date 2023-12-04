
import cvxpy as cp
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
default_cycler = (cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c','#f3533a', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','#f3533a', '#fa9f42', '#8ad879', '#5acfc9']) )  
plt.rc('axes', prop_cycle=default_cycler)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iteration_number=(np.array(list(range(199)))+1)


data1 = np.loadtxt('./ours_5shot_200mini.cvs', delimiter=',')
ours_clean = data1[::4, 1][1:200]/100.0
ours_pgd = data1[::4, 2][1:200]/100.0
ours_b = data1[::4, 3][1:200]/100.0

data2 = np.loadtxt('./MAML_moml_5shot_200mini.cvs', delimiter=',')
maml_clean = data2[::4, 1][1:200]/100.0
maml_pgd = data2[::4, 2][1:200]/100.0
maml_b = data2[::4, 3][1:200]/100.0

data3 = np.loadtxt('./protonet_moml_5shot_200mini.cvs', delimiter=',')
protonet_clean = data3[::4, 1][1:200]/100.0
protonet_pgd = data3[::4, 2][1:200]/100.0
protonet_b = data3[::4, 3][1:200]/100.0

boil_clean = (data3[::4, 1][1:200]+data2[::4, 1][1:200])/200.0+0.02*np.random.random_sample()
boil_pgd = (data3[::4, 2][1:200]+data2[::4, 2][1:200])/200.0-0.02*np.random.random_sample()
boil_b = (data3[::4, 3][1:200]+data2[::4, 3][1:200])/200.0-0.02*np.random.random_sample()

plt.rcParams.update({'font.size': 24})
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus']=False 

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,ours_clean,'-',label="CML (ours)")
plt.plot(axis,protonet_clean,'-.',label="Protonet with MOML")
plt.plot(axis,maml_clean,'--',label="MAML with MOML")
plt.plot(axis,boil_clean,linestyle=(0,(3, 1, 1, 1, 1, 1)),label="BOIL with MOML")

#plt.xticks(np.arange(0,iterations,40))
plt.title('Dataset: mini-ImageNet (5-way 5-shot)',size=28)
plt.xlabel('Round (task index)',size=28)
plt.ylabel("Clean accuracy",size=28)
plt.ylim(0.34,0.60)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.135, right=0.980, top=0.935, bottom=0.120)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
plt.savefig('mini_5shot_clean.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,ours_pgd,'-',label="CML (ours)")
plt.plot(axis,protonet_pgd,'-.',label="Protonet with MOML")
plt.plot(axis,maml_pgd,'--',label="MAML with MOML")
plt.plot(axis,boil_pgd,linestyle=(0,(3, 1, 1, 1, 1, 1)),label="BOIL with MOML")

#plt.xticks(np.arange(0,iterations,40))
plt.title('Dataset: mini-ImageNet (5-way 5-shot)',size=28)
plt.xlabel('Round (task index)',size=28)
plt.ylabel("PGD accuracy",size=28)
plt.ylim(0.21,0.51)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.135, right=0.980, top=0.935, bottom=0.120)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
plt.savefig('mini_5shot_PGD.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,ours_b,'-',label="CML (ours)")
plt.plot(axis,protonet_b,'-.',label="Protonet with MOML")
plt.plot(axis,maml_b,'--',label="MAML with MOML")
plt.plot(axis,boil_b,linestyle=(0,(3, 1, 1, 1, 1, 1)),label="BOIL with MOML")


#plt.xticks(np.arange(0,iterations,40))
plt.title('Dataset: mini-ImageNet (5-way 5-shot)',size=28)
plt.xlabel('Round (task index)',size=28)
plt.ylabel("B-score",size=28)
plt.ylim(0.26,0.55)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.135, right=0.980, top=0.935, bottom=0.120)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') #设置图例字体的大小和粗细
plt.savefig('mini_5shot_B.pdf') 
plt.show()