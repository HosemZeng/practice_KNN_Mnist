
'''
功能：
读取MNIST数据集并转换为numpy向量，每个numpy向量的元素代表一个图片向量或一个标签值
引用数据集链接：
http://yann.lecun.com/exdb/mnist/
引用代码链接：
https://blog.csdn.net/eleclike/article/details/79968574
'''

import numpy as np
import os

IMAGE_ROW = 28
IMAGE_COL = 28
IMAGE_SIZE = 28*28

def readHead(filename):
	dimension = []
	with open(filename,'rb') as pf:
		#获取magic number
		data = pf.read(4)#读出第1个4字节
		magic_num = int.from_bytes(data,byteorder='big')#bytes数据大尾端模式转换为int型
		
		#获取dimension的长度，由magic number的最后一个字节确定
		dimension_cnt = magic_num & 0xff
		
		#获取dimension数据，
		#dimension[0]表示图片的个数,如果是3维数据,dimension[1][2]分别表示其行/列数值
		for i in range(dimension_cnt):
			data = pf.read(4)
			dms = int.from_bytes(data,byteorder='big')
			dimension.append(dms)
	#print(dimension)
	return dimension


def getHeadLength(dimension):
	return 4*len(dimension)+4

	
def readImage(filename,head_len,offset):
	image = np.zeros((IMAGE_ROW,IMAGE_COL),dtype=np.uint8)#创建一个28x28的array，数据类型为uint8
	
	with open(filename,'rb') as pf:
		#magic_num的长度为4，dimension_cnt单个长度为4,前面的number个长度为28*28*offset	
		pf.seek(head_len+IMAGE_SIZE*offset) 
		
		for row in range(IMAGE_ROW):#处理28行数据，
			for col in range(IMAGE_COL):#处理28列数据
				data = pf.read(1)#单个字节读出数据
				pix = int.from_bytes(data,byteorder='big')#由byte转换为int类型，
				#简单滤波，如果该位置的数值大于指定值，则表示该像素为1.因为array已经初始化为0了，如果小于该指定值，不需要变化
				if pix >10:image[row][col] = 1
		#print(image)
	
	return image

def readLabel(filename,head_len,offset):
	label = None
	
	with open(filename,'rb') as pf:
		#pf 指向label的第number个数据,magic_num的长度为4，dimension_cnt单个长度为4
		pf.seek(head_len+offset) 
		data = pf.read(1)
		label = int.from_bytes(data,byteorder='big')#由byte转换为int类型，
	return label

def getSampleCount(dimension):
	return dimension[0]


def readImageVector(filename,head_len,offset,amount):
	image_mat=np.zeros((amount,IMAGE_SIZE),dtype=np.uint8)
	
	with open(filename,'rb') as pf:
		#magic_num的长度为4，dimension_cnt单个长度为4,前面的number个长度为28*28*offset	
		pf.seek(head_len+IMAGE_SIZE*offset) 
		
		for ind in range(amount):
			image = np.zeros((1,IMAGE_SIZE),dtype=np.uint8)#创建一个1，28x28的array，数据类型为uint8
			for row in range(IMAGE_SIZE):#处理28行数据，
				data = pf.read(1)#单个读出数据
				pix = int.from_bytes(data,byteorder='big')#由byte转换为int类型，
				#简单滤波，如果该位置的数值大于指定值，则表示该像素为1.因为array已经初始化为0了，如果小于该指定值，不需要变化
				if pix >10:image[0][row] = 1
			image_mat[ind,:]=image
			print('readImageVector：当前进度%0.2f%%'%(ind*100.0/amount),end='\r')
		print()
		#print(image)
	
	return image_mat

def readLabelVector(filename,head_len,offset,amount):
	label_list=[]
	
	with open(filename,'rb') as pf:
		#pf 指向label的第number个数据,magic_num的长度为4，dimension_cnt单个长度为4
		pf.seek(head_len+offset) 
		
		for ind in range(amount):
			data = pf.read(1)
			label = int.from_bytes(data,byteorder='big')#由byte转换为int类型，	
			label_list.append(label)
			print('readLabelVector：当前进度%0.2f%%'%(ind*100.0/amount),end='\r')
		print()
	
	return label_list

def readImageLabelVector(image_file,label_file,offset,amount):
	
	image_dim = readHead(image_file)
	label_dim = readHead(label_file)
	
	#判断样本中的image和label是否一致
	image_amount = getSampleCount(image_dim)
	label_amount = getSampleCount(label_dim)
	if image_amount != label_amount:
		print('Error:训练集image和label数量不相等')
		return None
	
	if offset+amount > image_amount:
		print('Error:请求的数据超出样本数量')
		return None
	
	#获取样本image和label的头文件长度
	image_head_len = getHeadLength(image_dim)
	label_head_len = getHeadLength(label_dim)
	
	#得到image和label的向量
	image_mat = readImageVector(image_file,image_head_len,offset,amount)
	label_list = readLabelVector(label_file,label_head_len,offset,amount)
	
	return image_mat,label_list