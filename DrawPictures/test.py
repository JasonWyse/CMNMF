
import numpy as np
import pylab as pl

import matplotlib.pyplot as plt

def unsupervised(evaluation_name,value):
	n_groups = 7
	#jaccard_value = (0.0329,0.0261,0.0259,0.0471,0.0509,0.0508,0.0551)
	fig,ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.4
	opacity = 0.4
	error_config = {'ecolor': '0.3'}
	rects1 = plt.bar(index, value, bar_width,
	                 alpha=opacity,
	                 color='b',
	                 #yerr=std_men,
	                 error_kw=error_config,
	                 label='Men')

	for tl in ax.get_xticklabels():
		tl.set_fontname("Times New Roman")

	plt.xlabel(u'Methods',fontname="Times New Roman",fontsize=18)#
	#pos=plt.get_position()
	#print pos
	plt.ylabel(evaluation_name+' Score',fontname="Times New Roman",fontsize=18)#
	plt.title(u'')
	plt.xticks(index + 0.2, ('HAC', 'Kmeans', 'KK', 'LDA', 'NMF', 'HMF','CMNMF'))
	#plt.legend()
	ax.set_ybound(0.5,1)

	#plt.tight_layout()
	plt.show()	
	fig.savefig(evaluation_name+'.eps',Format="eps")

def gama(evaluation_name,value):
	n_groups = 9
	#jaccard_value = (0.0329,0.0261,0.0259,0.0471,0.0509,0.0508,0.0551)
	fig,ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.4
	opacity = 0.4
	error_config = {'ecolor': '0.3'}
	rects1 = plt.bar(index, value, bar_width,
	                 alpha=opacity,
	                 color='b',
	                 #yerr=std_men,
	                 error_kw=error_config,
	                 label='Men')

	for tl in ax.get_xticklabels():
		tl.set_fontname("Times New Roman")

	plt.xlabel(u'gama value',fontname="Times New Roman",fontsize=18)#
	#pos=plt.get_position()
	#print pos
	plt.ylabel(evaluation_name,fontname="Times New Roman",fontsize=18)#
	plt.title(u'')
	plt.xticks(index + 0.2, ('0', '0.1','0.2','0.3','0.5', '0.7', '0.9', '1','1.5'))
	#plt.legend()
	ax.set_ybound(0.03,0.06)
	#plt.tight_layout()
	plt.show()	
	fig.savefig(evaluation_name+'.eps',Format="eps")

def supervised(evaluation_name,value1,value2,value3):
	n_groups = 5
	#jaccard_value = (0.0329,0.0261,0.0259,0.0471,0.0509,0.0508,0.0551)
	fig,ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.13
	opacity = 0.4
	error_config = {'ecolor': '0.3'}
	rects1 = plt.bar(index, value1, bar_width,
	                 alpha=opacity,
	                 color='b',
	                 #yerr=std_men,
	                 error_kw=error_config,
	                 label='Men')
	rects2 = plt.bar(index+bar_width, value2, bar_width,
	                 alpha=opacity,
	                 color='r',
	                 #yerr=std_men,
	                 error_kw=error_config,
	                 label='Men')
	rects3 = plt.bar(index+2*bar_width, value3, bar_width,
	                 alpha=opacity,
	                 color='g',
	                 #yerr=std_men,
	                 error_kw=error_config,
	                 label='Men')

	for tl in ax.get_xticklabels():
		tl.set_fontname("Times New Roman")

	plt.xlabel(u'roc n value',fontname="Times New Roman",fontsize=18)#
	#pos=plt.get_position()
	#print pos
	plt.ylabel('AUC',fontname="Times New Roman",fontsize=18)#
	plt.title(u'')
	plt.xticks(index + 0.2, ('10','50','100','150','230'))
	#plt.legend()
	ax.set_ybound(0,1)
	#plt.tight_layout()
	plt.show()	
	fig.savefig(evaluation_name+'.eps',Format="eps")

if __name__ == '__main__':
	#F1 = (0.0777,0.0662,0.0647,0.0979,0.093,0.0942,0.1044)
	#jaccard = (0.0404,0.0343,0.0334,0.0515,0.0488,0.0495,0.0551)
	# GO = (0.7742,0.7932,0.8148,0.8032,0.67,0.673,0.6226)
	# evaluation_name = 'GO'
	# unsupervised(evaluation_name,GO)
	#gamas = (0.0495,0.0518,0.0528,0.0535,0.0531,0.0543,0.0551,0.0545,0.0506)
	#gama('jaccard Score with different gama',gamas)
	hmf = (0.134,0.3533,0.5051,0.6060,0.7098)
	cmnmf = (0.1441,0.3558,0.5097,0.6119,0.7138)
	lp = (0.0487,0.1365,0.2524,0.3783,0.5417)
	supervised('AUC',lp,hmf,cmnmf)
	# #plt.rcParams['font.sans-serif'] = ['SimHei']
	# n_groups = 7
	# means_men = (20, 35, 30, 35, 27, 18)
	# #std_men = (2, 3, 4, 1, 2)
	# f1 = (0.0329,0.0261,0.0259,0.0471,0.0509,0.0508,0.0551)
	# means_women = (25, 32, 34, 20, 25, 18)
	# #std_women = (3, 5, 2, 3, 3)

	# fig, ax = plt.subplots()

	# index = np.arange(n_groups)
	# #bar_width = 0.35
	# bar_width = 0.4

	# opacity = 0.4
	# error_config = {'ecolor': '0.3'}

	# rects1 = plt.bar(index, f1, bar_width,
	#                  alpha=opacity,
	#                  color='b',
	#                  #yerr=std_men,
	#                  error_kw=error_config,
	#                  label='Men')


	# # def autolabel(rects):
	# #     # attach some text labels
	# #     for rect in rects:
	# #         height = rect.get_height()
	# #         ax.text(rect.get_x()+rect.get_width()/2.0, 1.05*height, 
	# #                 '%d'%int(height), ha='center', va='bottom')
	# # autolabel(rects1)
	# # autolabel(rects2)
	# # autolabel(rects3)
	# # autolabel(rects4)

	# for tl in ax.get_xticklabels():
	# 	pos1 = tl.get_position()
	# 	#print pos1
	# 	pos2 = (pos1[0]+1,pos1[1])
	# 	#print pos2
	# 	tl.set_position(pos2)
	# 	# tl.set_rotation(45)
	# 	#tl.set_fontsize(12)
	# 	tl.set_fontname("Times New Roman")
	# labels = [item.get_text() for item in ax.get_xticklabels()]
	# ax.set_xticklabels(labels)


	# plt.xlabel(u'Methods',fontname="Times New Roman",fontsize=18)#
	# #pos=plt.get_position()
	# #print pos
	# plt.ylabel(u'F1 Score',fontname="Times New Roman",fontsize=18)#
	# plt.title(u'')
	# plt.xticks(index + 0.2, ('HAC', 'Kmeans', 'KK', 'LDA', 'NMF', 'HMF','CMNMF'))
	# #plt.legend()
	# ax.set_ybound(0.02, 0.06)

	# #plt.tight_layout()
	# fig.bottom= 0.25
	# ax.bottom=0.25
	# plt.show()	
	# fig.savefig("F1.pdf",Format="PDF")