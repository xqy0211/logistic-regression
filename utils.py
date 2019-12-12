"""
filename:utils.py

@author:Cheng Zhuang
@date:2019/10/25 18:41
"""
import numpy as np
def loadDataSet():
	"""
	exam dataSet.

	x = [x1, x2]
	y in [0,1]  0=not admitted 1=admitted
	"""
	dataList = [] ; labelList = []
	with open('exam_x.dat') as fr:
		for line in fr.readlines():
			lineArr = line.strip().split()
			dataList.append([1.0, float(lineArr[0]), float(lineArr[1])])
	with open('exam_y.dat') as fr:
		for line in fr.readlines():
			lineArr = line.strip().split()
			labelList.append(float(lineArr[0]))

	return dataList, labelList

def normalize(dataList):
	"""
	Feature Scale.
	
	Args:
		dataList(List):
	
	Return:
		dataMat(matrix): data which have normalized.
		factor_scale(List): [xmin, xmax-xim, ymin, ymax-ymin]
	"""
	dataMat = np.mat(dataList); 
	xmin,ymin = np.min(dataMat[:,1]), np.min(dataMat[:,2])
	xmax,ymax = np.max(dataMat[:,1]), np.max(dataMat[:,2])
	dataMat[:,1] = (dataMat[:,1] - xmin) / (xmax - xmin)
	dataMat[:,2] = (dataMat[:,2] - ymin) / (ymax - ymin)
	factor_scale = [xmin, xmax-xmin, ymin, ymax-ymin]
	return dataMat, factor_scale

def store(value, filename):
	"""
	store value to file.
	"""
	import pickle
	with open(filename, 'wb') as fw:
		pickle.dump(value, fw)

def grab(filename):
	"""
	read value from file.
	"""
	import pickle
	with open(filename, 'rb') as fr:
		value = pickle.load(fr)
	return value

if __name__ == "__main__":
	dataList, labelList = loadDataSet()
	print(normalize(dataList)[0])