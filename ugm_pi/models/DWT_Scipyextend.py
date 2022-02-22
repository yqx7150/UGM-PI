import scipy.misc as sm
import numpy as np
#import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from scipy.io import loadmat
import glob
import scipy.io as io
try:
    from skimage.measure import compare_psnr,compare_ssim,compare_mse
except:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr,structural_similarity as compare_ssim,mean_squared_error as compare_mse


def DWT(data_1):
	ld = np.array([[ 0.707106781186548, 0.707106781186548]],dtype=np.float64)
	hd = np.array([[ -0.707106781186548, 0.707106781186548]],dtype=np.float64)
	ld_T = np.array([[0.707106781186548], [0.707106781186548]],dtype=np.float64)
	hd_T = np.array([[-0.707106781186548], [0.707106781186548]],dtype=np.float64)

	lr = np.array([[ 0.707106781186548, 0.707106781186548]],dtype=np.float64)
	hr = np.array([[0.707106781186548, -0.707106781186548]],dtype=np.float64)
	lr_T = np.array([[ 0.707106781186548], [0.707106781186548]],dtype=np.float64)
	hr_T = np.array([[0.707106781186548], [-0.707106781186548]],dtype=np.float64)

	GS = convolve2d(data_1,hd,boundary='symm')
	HS = convolve2d(data_1,ld,boundary='symm') 
	WH = convolve2d(GS,ld_T,boundary='symm') 
	WW = convolve2d(GS,hd_T,boundary='symm') 
	HW = convolve2d(HS,hd_T,boundary='symm') 
	S = convolve2d(HS,ld_T,boundary='symm')
	'''
	WH = loadmat('./dealwith/WH.mat')['WH']
	WW = loadmat('./dealwith/WW.mat')['WW']
	HW = loadmat('./dealwith/HW.mat')['HW']
	S = loadmat('./dealwith/S.mat')['S']
	plt.figure(1)
	plt.subplot(2,2,1)
	plt.imshow(S,cmap='gray')
	plt.subplot(2,2,2)
	plt.imshow(HW,cmap='gray')
	plt.subplot(2,2,3)
	plt.imshow(WH,cmap='gray')
	plt.subplot(2,2,4)
	plt.imshow(WW,cmap='gray')
	plt.show()
	'''
	return WH,WW,HW,S

def iDWT(WH,WW,HW,S):
	ld = np.array([[ 0.707106781186548, 0.707106781186548]],dtype=np.float64)
	hd = np.array([[ -0.707106781186548, 0.707106781186548]],dtype=np.float64)
	ld_T = np.array([[0.707106781186548], [0.707106781186548]],dtype=np.float64)
	hd_T = np.array([[-0.707106781186548], [0.707106781186548]],dtype=np.float64)

	lr = np.array([[ 0.707106781186548, 0.707106781186548]],dtype=np.float64)
	hr = np.array([[0.707106781186548, -0.707106781186548]],dtype=np.float64)
	lr_T = np.array([[ 0.707106781186548], [0.707106781186548]],dtype=np.float64)
	hr_T = np.array([[0.707106781186548], [-0.707106781186548]],dtype=np.float64)
	lf = len(lr)


	GS = 0.5 * (convolve2d(WH,lr_T,boundary='symm') + convolve2d(WW,hr_T,boundary='symm'))
	#print('1',GS.shape)
	[nr,nc]=GS.shape
	#print('20',GS.shape)
	GS = GS[0+lf:nr-lf,:]
	#print('2',GS.shape)
	HS = 0.5 * (convolve2d(S,lr_T,boundary='symm') + convolve2d(HW,hr_T,boundary='symm'))
	#print('1',HS.shape)
	HS = HS[lf:nr-lf,:];
	#print('2',HS.shape)
	img_r = 0.5 * (convolve2d(HS,lr,boundary='symm') + convolve2d(GS,hr,boundary='symm'))
	#print('1',img_r.shape)
	[nr,nc] = img_r.shape;
	img_r = img_r[:,lf:nc-lf];
	#img_r = img_r[2:258,2:258];
	
	img_rec=img_r
	return img_rec
	
	'''
	print(np.max(img_r),np.min(img_r))
	psnr = compare_psnr(img_r,data_10)
	ssim = compare_ssim(img_r,data_10)
	print('psnr',psnr,ssim)
	error = np.mean((np.mean((img_r - data_10)**2)))
	print('error',error)
	'''


	  
