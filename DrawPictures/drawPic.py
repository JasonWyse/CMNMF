from __future__ import division
import sys
import os
import re
import math
from numpy import *
from path import PathManager
import evaluate
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
import colorsys
import  numpy as np
def rand_hsl():
    '''Generate a random hsl color.'''
    h = random.uniform(0.02, 0.31) + random.choice([0, 1/3.0,2/3.0])
    l = random.uniform(0.3, 0.8)
    s = random.uniform(0.3, 0.8)

    rgb = colorsys.hls_to_rgb(h, l, s)
    return (int(rgb[0]*256), int(rgb[1]*256), int(rgb[2]*256))
def bar_chart_generator(data_90,data_100):
	n_groups = len(data_90)      
	working_data_90 = data_90
	working_data_100 = data_100 
    
	fig, ax = plt.subplots()  
	index = np.arange(n_groups)  
	print index
	bar_width = 0.35  
   
	opacity = 0.4  
	color1 = rand_hsl()
	color2 = rand_hsl()
	rects1 = plt.bar(index, working_data_90, bar_width,alpha=opacity, color='b',label = '90 percent')# label = '' 
	rects2 = plt.bar(index + bar_width, working_data_100, bar_width,alpha=opacity,color='y',label='100 percent')  #
   
	plt.xlabel('Query')  
	plt.ylabel('% of dimension')  
	plt.title('Percent of dimension by query')  
	#plt.xticks(index + bar_width, ('A', 'B', 'C', 'D', 'E'))  
	plt.ylim(0,1.17)  
	plt.legend()     
	plt.tight_layout()  
	plt.show()
	#saveName = 'percentDimension_q20'
	#fig.savefig(saveName,format="eps")
def bar_chart_generator1(data_90,bar_width):
	n_groups = len(data_90)
	working_data_90 = data_90
	working_data_100 = data_100 
    
	fig, ax = plt.subplots()  
	index = [0,0.05,0.10,0.15,0.20,0.25,0.30,0.35] 
	print index
	#bar_width = 0.35  
   
	opacity = 0.4  
	rects1 = plt.bar(index, working_data_90, bar_width,alpha=opacity, color='b',label = '90 percent')# label = '' 
	#rects2 = plt.bar(index + bar_width, working_data_100, bar_width,alpha=opacity,color='y',label='100 percent')  #
   
	plt.ylabel('Query %')  
	plt.xlabel('% of dimension')  
	plt.title('Percent of dimension by query')  
#	plt.xticks(index + bar_width, ('0', '0.05', '0.10', '0.25', '0.30','0.35'))  
	#plt.ylim(0,1.17)  
	plt.legend()     
	plt.tight_layout()  
	plt.show()
	#saveName = 'percentDimension_q20'
	#fig.savefig(saveName,format="eps")
def hist_generator1(data_90,bin_s,title,xtick_base,interval):
	n_groups = len(data_90)      
	working_data_90 = data_90	   
	working_data_100 = data_100
	#bins = np.linspace(0, 0.40, 10)
   
	x_ticks = [] 

	for i in range(len(bin_s)):
		x_ticks.append(str(interval*i+xtick_base))
	plt.hist(working_data_90,bin_s)#,normed = True, stacked = True
	plt.xticks(bin_s ,x_ticks)
	#plt.yticks(bin_s, x_ticks);
	plt.xlabel('% of dimension')  
	plt.ylabel('# queries')  
	#title = 'Query distribution when taking 90% energy'
	plt.title(title)  
	plt.show()
def hist_generator2():
	x = np.linspace(0,1,10)
	y = [16,29,16,1,0,0,0,0,0,0]	
	#y2 = [0,0,0,0,1,0,0,1,6,54]
	width = 0.035
	xticklabels = []
	#yticklabels_y2 = [0,10,20,30,40,50,60]
	yticklabels_y = [0,5,10,15,20,25,30]
	for i in range(10):
		xticklabels.append('%s - %s'%(format(0.1*i,'2.0%'),format(0.1*(i+1),'2.0%')))
	#plt.figure(figsize=(8,6),dpi=120)
	fig, ax_all = plt.subplots(nrows = 1,ncols = 1)#subplot(numRows, numCols, plotNum)
	ax = ax_all
	#ax2 = ax_all[1]
	# fig1 = fig[0]
	# fig2 = fig[1]
	ax.set_xticks(x)
	# ax.title('(a) Capturing 90% energy')
	
	# ax2.title('(b) Capturing 100% energy')
	#fig2,ax2 = plt.subplots(212)
	rects0 = ax.bar(x,y,width,color = ['b'])
	ax.set_yticklabels(yticklabels_y)
	for tl in ax.get_yticklabels():
		
		#tl.set_fontsize(12)
		tl.set_fontname("Times New Roman")
	
	ax.set_xticklabels(xticklabels)
	
	#plt.caption('slk')
	#ax2.set_xticklabels(xticklabels)
	pos = ax.get_position()
	#ax.set_position(matplotlib.transforms.Bbox(array([pos[0]+0.1,pos[1]+0.1])))
	#ax.set_position(pos[0]+0.1,pos[1]+0.1)
	
	for tl in ax.get_xticklabels():
		pos1 = tl.get_position()
		pos2 = (pos1[0]-0.02,pos1[1])
		tl.set_position(pos2)
		tl.set_rotation(45)
		#tl.set_fontsize(12)
		tl.set_fontname("Times New Roman")
	
	plt.xlabel('Percetage range',fontname="Times New Roman")  
	plt.ylabel('# queries',fontname="Times New Roman") 
	#plt.title('capture')
	plt.show()
def hist_generator3():
	x = np.linspace(0,1,10)
	y = [16,29,16,1,0,0,0,0,0,0]	
	y2 = [0,0,0,0,1,0,0,1,6,54]
	width = 0.035
	xticklabels = []
	yticklabels_y2 = [0,10,20,30,40,50,60]
	yticklabels_y = [0,5,10,15,20,25,30]
	for i in range(10):
		xticklabels.append('%s - %s'%(format(0.1*i,'2.0%'),format(0.1*(i+1),'2.0%')))
	#plt.figure(figsize=(8,6),dpi=120)
	fig, ax_all = plt.subplots(nrows = 1,ncols = 1)#subplot(numRows, numCols, plotNum)
	ax = ax_all
	#ax2 = ax_all[1]
	# fig1 = fig[0]
	# fig2 = fig[1]
	ax.set_xticks(x)
	# ax.title('(a) Capturing 90% energy')
	
	# ax2.title('(b) Capturing 100% energy')
	#fig2,ax2 = plt.subplots(212)
	rects0 = ax.bar(x,y2,width,color = ['b'])
	ax.set_yticklabels(yticklabels_y2)
	#ax.set_yticklabels(yticklabels_y)
	for tl in ax.get_yticklabels():
		
		tl.set_fontsize(12)
		tl.set_fontname("Times New Roman")
	
	ax.set_xticklabels(xticklabels)
	# plt.sca(ax)
	# plt.xlabel('Percetage range')  
	# plt.ylabel('# queries') 
	# plt.title('(a) Capturing 90% energy')
	# plt.sca(ax2)
	# plt.xlabel('Percetage range')  
	# plt.ylabel('# queries') 
	# plt.title('(b) Capturing 100% energy')
	# ax2.set_xticks(x)
	# rects1 = ax2.bar(x,y2,width,color = ['b'])
	# plt.xlabel('Percetage range')  
	# plt.ylabel('# queries') 
	# plt.title('(b)')
	#xticklabels = ['0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5',]	
	ax.set_xticklabels(xticklabels)
	
	#plt.caption('slk')
	#ax2.set_xticklabels(xticklabels)
	pos = ax.get_position()
	#ax.set_position(matplotlib.transforms.Bbox(array([pos[0]+0.1,pos[1]+0.1])))
	#ax.set_position(pos[0]+0.1,pos[1]+0.1)
	
	for tl in ax.get_xticklabels():
		pos1 = tl.get_position()
		pos2 = (pos1[0]-0.02,pos1[1])
		tl.set_position(pos2)
		tl.set_rotation(45)
		tl.set_fontsize(12)
		tl.set_fontname("Times New Roman")
	# for tl in ax2.get_xticklabels():
	# 	pos1 = tl.get_position()
	# 	pos2 = (pos1[0]-0.02,pos1[1])
	# 	tl.set_position(pos2)
	# 	tl.set_rotation(45)
	# 	tl.set_fontsize(10)
	# pos1 = ax2.get_position()
	# print pos1
#	ax2.set_position(pos1[0],pos1[1]-1)
	plt.xlabel('Percetage range',fontname="Times New Roman",fontsize="12")  
	plt.ylabel('# queries',fontname="Times New Roman",fontsize="12") 
	#plt.title('capture')
	plt.show()
def hist_generator(data_90,data_100,bins,title,xtick_base,interval):
	n_groups = len(data_90)      
	working_data_90 = data_90	   
	working_data_100 = data_100
	#bins = np.linspace(0, 0.40, 10)
   
	x_ticks = [] 

	for i in range(len(bins)):
		x_ticks.append(str(interval*i+xtick_base))
	plt.hist(working_data_90,bins)
	plt.hist(working_data_100,bins)
	plt.xticks(bins , x_ticks)
	plt.xlabel('% of dimension')  
	plt.ylabel('# queries')  
	#title = 'Query distribution when taking 90% energy'
	plt.title(title)  
	plt.show()	
def test_performance_pic(datafile):
	data = np.loadtxt(datafile)
	RSVM_index = 6
	RankNet_index = 5
	ListNet_index = 3
	FacRSVM_index = 2

	rankSVM = data[RSVM_index-1]
	print rankSVM
	return

if __name__ == '__main__':
	"""
	# if len(sys.argv) < 2:
	# 	sys.exit(-1)
	# filename = sys.argv[1]	
	# DT={'0':'Mq2007','1':'Mq2008','2':'OHSUMED'}
	
	# #ipos2={'0':'test','1':'vali','2':'train'}
	# iMaxPosition=10
	# dataset = DT.get(sys.argv[1])	
	# #dataset=DT.get(raw_input('pls choose dataset(0:Mq2007,1:Mq2008,2:OHSUMED):\n'))
	# #msr=Msr.get(raw_input('pls choose measure (0:MAP,1:NDCG):\n'))	
	# algthm_name='svm_light2'
	# p=PathManager(dataset,algthm_name)
	# pth_bslns=p.getPath('path_baselines')
	# pth_evlt=p.getPath('path_evaluation')
	# pth_dataset=p.getPath('path_dataset')
	# pth_data=p.getPath('path_data')

	# for fold_number in [1]:#,2,3,4,5
		# train= os.path.join(pth_dataset,'Fold%d'%fold_number,'train.txt')
		# train_pp =  os.path.join(pth_dataset,'Fold%d'%fold_number,'train_pp.txt')
		# test = os.path.join(pth_dataset,'Fold%d'%fold_number,'test.txt')
		# vali = os.path.join(pth_dataset,'Fold%d'%fold_number,'vali.txt')
		# model_file = os.path.join(pth_dataset,'Fold%d'%fold_number,'svm_train_model')

		# qid_pos_percent_file = os.path.join(pth_dataset,'Fold%d'%fold_number,'qid_pos_percent.txt')
		# qid_pos_percent_100_file = os.path.join(pth_dataset,'Fold%d'%fold_number,'qid_pos_percent_100.txt')
		
		# data_90 = np.loadtxt(qid_pos_percent_file)
		# data_100 = np.loadtxt(qid_pos_percent_100_file)
 
# # plot the first column as x, and second column as y
		# k_queries = len(data_90)

# #		bar_chart_generator(data_90[0:k_queries],data_100[0:k_queries])
		# interval = 0.1
		# title = 'Query distribution when capturing 90% energy'
		# bins = np.linspace(0, 1, 11)
		# #print bins
		# xtick_base = 0
		#hist_generator1(data_90[0:k_queries],bins,title,xtick_base,interval)

		# bar_width = 0.05
		# index = [0,0]
		# array = [8,8,15,14,10,6,1]
		# data= array/sum(array)
		# print 'data',data
		# bar_chart_generator1(data,bar_width)
		#bar_chart_generator(data_90,data_100)
		
		# title = 'Query distribution when taking 100% energy'
		# bins = np.linspace(0.68, 1, 9)
		# print bins
		# xtick_base = 0.68
		# hist_generator(data_100[0:k_queries],bins,title,xtick_base,interval);
		
		# interval = 0.1
		# title = 'Query distribution'
		# bins = np.linspace(0, 1, 11)
		# print bins
		# xtick_base = 0
		# hist_generator(data_90,data_100,bins,title,xtick_base,interval)
	#	hist_generator(data_100[0:k_queries],bins,title,xtick_base,interval);
#	datafile = 'ohsumed_test_performance'
#	test_performance_pic(datafile)
	"""
	#hist_generator2()
	hist_generator3()