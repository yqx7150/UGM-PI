import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import glob
import scipy.io as io
from ugm_pi.models.DWT_Scipyextend import *
from ugm_pi.models.cond_refinenet_dilated import CondRefineNetDilated
try:
    from skimage.measure import compare_psnr,compare_ssim,compare_mse
except:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr,structural_similarity as compare_ssim,mean_squared_error as compare_mse

import time
__all__ = ['SIAT_iOrth_8h_10sigma']


def ReadMat(path):
    try:
        mat = io.loadmat(path)
    except:
        import scipy.io as io
        mat = io.loadmat(path)
    for i in mat.keys():
        if '__' not in i:
            return mat[i]


def compare_hfen(ori, rec):
    operation = np.array(io.loadmat("./loglvbo.mat")['h1'], dtype=np.float32)
    ori = cv2.filter2D(ori.astype('float32'), -1, operation, borderType=cv2.BORDER_CONSTANT)
    rec = cv2.filter2D(rec.astype('float32'), -1, operation, borderType=cv2.BORDER_CONSTANT)
    hfen = np.linalg.norm(ori-rec, ord='fro')
    return hfen


class SIAT_iOrth_8h_10sigma():
    def __init__(self, args, config):
        self.args = args
        self.config = config
    def test(self):
        checkpoint_num = 90000
        states0 = torch.load(os.path.join(self.args.log, 'checkpoint_' + str(checkpoint_num)+ '.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states0[0])
        scorenet.eval()
        with torch.no_grad():
            data_list = glob.glob('./input_data/data/*.mat')
            for data_path in data_list:
                data_name = data_path.split('/')[-1].split('.')[0]
                ori_data = ReadMat(data_path)                
                ori_data_tmp = ori_data/np.max(np.abs(ori_data))
                sos_ori_data = np.sqrt(np.sum(np.square(np.abs(ori_data_tmp)),axis=2))
                nor_sos_ori_data = sos_ori_data/np.max(sos_ori_data)
                coils = ori_data.shape[-1]
                mask_list = glob.glob('./input_data/mask/*.mat')
                for mask_path in mask_list:
                    mask = ReadMat(mask_path)
                    mask = np.fft.fftshift(mask)
                    mask = mask[...,None].repeat(coils,-1)
                    Sample_rate = round(np.sum(mask[...,0]) / (mask.shape[1] ** 2), 3)
                    data = ori_data/np.max(np.abs(ori_data))
                    kdata = np.fft.fftn(data,axes=[0,1])
                    ksample = kdata * mask
                    ksample = ksample.transpose(2,0,1)
                    
                    x = nn.Parameter(torch.Tensor(coils,8,258,258).uniform_(-1,1)).cuda()
                    x1 = x.clone()
                    step_lr=0.5*0.00002  
                    sigmas = np.exp(np.linspace(np.log(1.0), np.log(0.01),10))
                    max_psnr = 0
                    psnr = 0
                    start = time.time()
                    for idx, sigma in enumerate(sigmas):
                        print('current sigma is :{}\t,R:{}\t'.format(idx, round(Sample_rate,4)))
                        labels  = (torch.ones(1, device=x.device) * idx).long()
                        step_size = step_lr * (sigma / sigmas[-1]) ** 2
                        print('sigma = {},data: {}'.format(sigma,data_name))
                        n_steps_each = 30
                        for step in range(n_steps_each):
                            inner = time.time()
                            noise = torch.rand_like(x)* np.sqrt(step_size * 2)
                            x = x.cuda()
                            noise = noise.cuda()
                            grad= scorenet(x1, labels).detach()
                            x = x + step_size * grad.cuda()
                            x1 = x + noise
                            x=np.array(x.cpu().detach(),dtype = np.float32)
                            x = x[:,:,:257,:257]
                            img_r = np.zeros([coils,256,256])
                            img_i = np.zeros([coils,256,256])
                            img_rec = np.zeros([coils,256,256]).astype(np.complex64)
                            img_rec_Kdata3 = np.zeros([256,256,coils]).astype(np.complex64)
                            img_rec_Kdata = np.zeros([256,256,coils]).astype(np.complex64)

                            for imageindex in range(coils):
                                img_r[imageindex,:,:]=iDWT(
                                    x[imageindex,0,:,:],
                                    x[imageindex,1,:,:],
                                    x[imageindex,2,:,:],
                                    x[imageindex,3,:,:])
                                img_r[imageindex,:,:]=img_r[imageindex,:,:].real
                                img_i[imageindex,:,:]=iDWT(
                                    x[imageindex,4,:,:],
                                    x[imageindex,5,:,:],
                                    x[imageindex,6,:,:],
                                    x[imageindex,7,:,:])
                                img_i[imageindex,:,:]=img_i[imageindex,:,:].real

                                img_rec[imageindex,:,:] = img_r[imageindex,:,:] +1j*img_i[imageindex,:,:]
                                img_rec_Kdata[:,:,imageindex]=np.fft.fft2(img_rec[imageindex,:,:])
                                One=np.ones([256,256])
                                img_rec_Kdata2=img_rec_Kdata[:,:,imageindex]*(One-mask[:,:,imageindex])+ksample[imageindex,:,:]
                                img_rec_Kdata3[:,:,imageindex] = img_rec_Kdata2
                                img_rec[imageindex,:,:]=np.fft.ifft2(img_rec_Kdata3[:,:,imageindex])
                                img_r[imageindex,:,:] = img_rec[imageindex,:,:].real
                                img_i[imageindex,:,:] = img_rec[imageindex,:,:].imag
                                WH_r,WW_r,HW_r,S_r=DWT(img_r[imageindex,:,:])
                                
                                x[imageindex,0,:,:]=WH_r
                                x[imageindex,1,:,:]=WW_r
                                x[imageindex,2,:,:]=HW_r
                                x[imageindex,3,:,:]=S_r

                                WH_i,WW_i,HW_i,S_i=DWT(img_i[imageindex, :, :])    
                                x[imageindex,4,:,:]=WH_i
                                x[imageindex,5,:,:]=WW_i
                                x[imageindex,6,:,:]=HW_i
                                x[imageindex,7,:,:]=S_i
                            

                            temp=np.zeros([coils,8,258, 258])
                            temp[...,:257,:257]=x

                            x=torch.tensor(temp,dtype=torch.float32).cuda()

                            sos_rec = np.sqrt(np.sum(np.square(np.abs(img_rec)),axis=0))
                            nor_sos_ori_data = sos_ori_data/np.max(sos_ori_data)
                            nor_sos_rec = sos_rec/np.max(sos_rec)
                            psnr = compare_psnr(np.abs(nor_sos_rec),np.abs(nor_sos_ori_data),data_range=1)
                            ssim = compare_ssim(np.abs(nor_sos_rec), np.abs(nor_sos_ori_data), data_range=1)
                            hfen = compare_hfen(np.abs(nor_sos_rec), np.abs(nor_sos_ori_data))
                            if max_psnr<psnr:
                                max_psnr = psnr
                                current_ssim = ssim
                                current_hfen = hfen
                            print("step:{}\tPSNR :{:.3f}\tSSIM :{:.3f}\tHFEN :{:.3f}\t iter time:{:.2f}s".format(step,psnr,ssim,hfen,time.time() - inner))
                    end = time.time() - start
                    print('Max PSNR:{:.3f}\tMax SSIM:{:.3f}\tMax HFEN:{:.3f}\tcost time:{:.2f}s'.format(max_psnr,current_ssim,current_hfen, end))
