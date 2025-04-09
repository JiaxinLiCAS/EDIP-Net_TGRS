# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug
空天信息创新研究院20-25直博生，导师高连如

"""
"""
❗❗❗❗❗❗#此py作用：第四阶段的决策融合
"""

from .evaluation import MetricsCal
import os
import scipy.io as sio
import numpy as np


def compute_psnr(x_true, x_pred):
    assert x_true.ndim == 3 and x_pred.ndim ==3

    img_w, img_h, img_c = x_true.shape
    ref = x_true.reshape(-1, img_c)
    #print(ref)
    tar = x_pred.reshape(-1, img_c)
    msr = np.mean((ref - tar)**2, 0) #列和
    #print(msr)
    max2 = np.max(ref,0)**2
    #print(max2)
    psnrall = 10*np.log10(max2/msr)
    #print(psnrall)
    m_psnr = np.mean(psnrall)
    #print(m_psnr)
    psnr_all = psnrall.reshape(img_c)
    #print( psnr_all)
    return m_psnr,psnr_all

def compute_rmse_byband(x_true, x_pre):
     assert x_true.ndim == 3 and x_pre.ndim ==3 and x_true.shape == x_pre.shape
     img_w, img_h, img_c = x_true.shape
     ref = x_true.reshape(-1, img_c)
     tar = x_pre.reshape(-1, img_c)
     rmse_byband=np.sqrt(np.mean((ref - tar)**2, 0))
     return rmse_byband

def compute_sammap(x_true, x_pred):
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape
    w, h, c = x_true.shape
    x_true = x_true.reshape(-1, c) #一行为一条光谱曲线
    x_pred = x_pred.reshape(-1, c)

    #sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)+1e-5) 原本的
    #sam_all  = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))
    sam_all = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)+1e-7)
    sam_all = np.arccos(sam_all) * 180 / np.pi
    sammap  = sam_all.reshape(w, h)
    return  sammap

def compute_rmsemap(x_true, x_pred):
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape
    w, h, c = x_true.shape
    x_true = x_true.reshape(-1, c) #一行为一条光谱曲线
    x_pred = x_pred.reshape(-1, c)

    #sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1)+1e-5) 原本的
    #sam_all  = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))
    rmse_all = np.sqrt(np.mean((x_true - x_pred)**2, 1))
    rmsemap=rmse_all.reshape(w, h)
    return  rmsemap
            


def decision_srf(Out_fhsi,Out_fmsi,blind,decide='rmse'):
    
    Out_fhsi_numpy = Out_fhsi.data.cpu().numpy()[0].transpose(1,2,0) #dtype('float32')
    Out_fmsi_numpy = Out_fmsi.data.cpu().numpy()[0].transpose(1,2,0)
    
   
    gt=blind.gt
    hr_msi=blind.tensor_hr_msi.data.cpu().numpy()[0].transpose(1,2,0) # H W C
    w,h,c = gt.shape
    
    
    #估计的srf
    srf = blind.model.srf #  torch.Size([8, 191, 1, 1])
    srf_est=np.squeeze(srf.data.cpu().detach().numpy()).T #(191, 8)                   

    '''
    #####计算SRF对two candidates生成HrMSI与真值的差距####
    '''
    if srf_est.shape[0] == c:
        #fhsi
        hr_msi_est_fhsi_numpy = np.dot(Out_fhsi_numpy.reshape(w*h,c), srf_est).reshape(w,h,srf_est.shape[1]) #(300, 300, 8) numpy
        #fmsi
        hr_msi_est_fmsi_numpy = np.dot(Out_fmsi_numpy.reshape(w*h,c), srf_est).reshape(w,h,srf_est.shape[1]) #(300, 300, 8) numpy
        
    #print('Out_fhsi_numpy dtype {}'.format(Out_fhsi_numpy.dtype))


    if decide!='rmse':
        ####计算sammap####
        sammap_fhsi =compute_sammap(hr_msi, hr_msi_est_fhsi_numpy)
        sammap_fmsi =compute_sammap(hr_msi, hr_msi_est_fmsi_numpy)
        
        #sammap_fhsi_gt =compute_sammap(gt, Out_fhsi_numpy)
        #sammap_fmsi_gt =compute_sammap(gt, Out_fmsi_numpy)
        
        ##根据hrmsi每个像素的sam 选择hrhsi对应像素曲线##
        sammap_diff=sammap_fhsi-sammap_fmsi
        sammap_diff_mark=np.zeros(sammap_diff.shape)
        sammap_diff_mark[sammap_diff<=0]=1 #该像素fhsi重建质量更好
        sammap_diff_mark[sammap_diff>0]=2  #该像素fmsi重建质量更好
        
        hr_hsi_srf_sam=np.ones(gt.shape)
        
        for i in range(w):
            for j in range(h):
                if sammap_diff_mark[i,j]==1:
                    hr_hsi_srf_sam[i,j,:]=Out_fhsi_numpy[i,j,:]
                if sammap_diff_mark[i,j]==2:
                    hr_hsi_srf_sam[i,j,:]=Out_fmsi_numpy[i,j,:]
        

        sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(gt,hr_hsi_srf_sam, blind.args.scale_factor)
        L1=np.mean( np.abs( gt- hr_hsi_srf_sam ))
        information3_srf="gt与hr_hsi_srf_sam\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
        print(information3_srf)
        ##根据hrmsi每个像素的sam 选择hrhsi对应像素曲线##        
             
        
        #保存精度
        file_name = os.path.join(blind.args.expr_dir, 'Stage4.txt')
        with open(file_name, 'a') as opt_file:
            
            #opt_file.write('--------------decision----------------')
            #opt_file.write('\n')
            opt_file.write(information3_srf)
            opt_file.write('\n')
            #opt_file.write(information4_srf)
            #opt_file.write('\n')
        
    if decide=='rmse':
        
        ####计算sammap####
        rmsemap_fhsi =compute_rmsemap(hr_msi, hr_msi_est_fhsi_numpy)
        rmsemap_fmsi =compute_rmsemap(hr_msi, hr_msi_est_fmsi_numpy)
        
       
        
        ##根据hrmsi每个像素的rmse 选择hrhsi对应像素曲线##
        rmsemap_diff=rmsemap_fhsi-rmsemap_fmsi
        rmsemap_diff_mark=np.zeros(rmsemap_diff.shape)
        rmsemap_diff_mark[rmsemap_diff<=0]=1 #该像素fhsi重建质量更好
        rmsemap_diff_mark[rmsemap_diff>0]=2  #该像素fmsi重建质量更好 和sammap_diff_mark结果不一样
        
        hr_hsi_srf_rmse=np.ones(gt.shape) #dtype float64
        
        for i in range(w):
            for j in range(h):
                if rmsemap_diff_mark[i,j]==1:
                    hr_hsi_srf_rmse[i,j,:]=Out_fhsi_numpy[i,j,:]
                if rmsemap_diff_mark[i,j]==2:
                    hr_hsi_srf_rmse[i,j,:]=Out_fmsi_numpy[i,j,:]
        
        #print('Out_fhsi_numpy dtype {}'.format(Out_fhsi_numpy.dtype))


        sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(gt,hr_hsi_srf_rmse, blind.args.scale_factor)
        L1=np.mean( np.abs( gt- hr_hsi_srf_rmse ))
        information5_srf="gt与hr_hsi_srf_rmse逐像素\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
        print(information5_srf)
        ##根据hrmsi每个像素的rmse 选择hrhsi对应像素曲线##        

        
        #保存精度
        file_name = os.path.join(blind.args.expr_dir, 'Stage4.txt')
        with open(file_name, 'a') as opt_file:
            
            #opt_file.write('--------------decision----------------')
           
            opt_file.write(information5_srf)
            opt_file.write('\n')
          
    
    if decide!='rmse':
        return hr_hsi_srf_sam
    else:
        return hr_hsi_srf_rmse, rmsemap_fhsi, rmsemap_fmsi,rmsemap_diff_mark #返回三维tensor H W C
    
def select_decision(Out_fhsi,Out_fmsi,blind,decide): #Out_fhsi,Out_fmsi是四维device tensor
    

    
    #if blind.args.select == True:
    '''决策融合'''
    
    srf_out,rmsemap_fhsi,rmsemap_fmsi,rmsemap_diff_mark=decision_srf(Out_fhsi,Out_fmsi,blind,decide) #返回三维tensor 不在device
        
    #保存决策结果
    
    sio.savemat(os.path.join(blind.args.expr_dir, 'srf_Out_S4.mat'), {'Out':srf_out})

   
    '''决策融合'''

        
        
  

    return srf_out
    
   
    
    
    
    

