# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug
空天信息创新研究院20-25直博生，导师高连如

"""
"""
❗❗❗❗❗❗#此py作用：对应第三阶段的图像生成
"""
import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import scipy
import torch.nn.functional as fun

from .evaluation import MetricsCal

from .network_s3 import double_U_net_skip


''' PSF and SRF '''    
class PSF_down():
    def __call__(self, input_tensor, psf, ratio): #PSF为#1 1 ratio ratio 大小的tensor
            _,C,_,_=input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2],input_tensor.shape[3]
            if psf.shape[0] == 1:
                psf = psf.repeat(C, 1, 1, 1) #8X1X8X8
                                                   #input_tensor: 1X8X400X400
            output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio),  groups=C) #ratio为步长 None代表bias为0，padding默认为无
            return output_tensor
    
class SRF_down(): 
    def __call__(self, input_tensor, srf): # srf 为 ms_band hs_bands 1 1 的tensor      
            output_tensor = fun.conv2d(input_tensor, srf, None)
            return output_tensor
        
class dip():
    def __init__(self,args,Out_fhsi,Out_fmsi,psf,srf,blind):
     
        assert(Out_fhsi.shape == Out_fmsi.shape)
        
        #获取SRF and PSF
        
        self.Out_fhsi=Out_fhsi
        self.Out_fmsi=Out_fmsi
        
        self.args=args
        
        self.hr_msi=blind.tensor_hr_msi #四维
        self.lr_hsi=blind.tensor_lr_hsi #四维
        self.gt=blind.gt #三维
        
        psf_est = np.reshape(psf, newshape=(1, 1, self.args.scale_factor, self.args.scale_factor)) #1 1 ratio ratio 大小的tensor
        self.psf_est = torch.tensor(psf_est).to(self.args.device).float()
        srf_est = np.reshape(srf.T, newshape=(srf.shape[1], srf.shape[0], 1, 1)) #self.srf.T 有一个T转置 (8, 191, 1, 1)
        self.srf_est = torch.tensor(srf_est).to(self.args.device).float()             # ms_band hs_bands 1 1 的tensor torch.Size([8, 191, 1, 1])
        
        self.psf_down=PSF_down() #__call__(self, input_tensor, psf, ratio):
        self.srf_down=SRF_down() #__call__(self, input_tensor, srf):
            
        self.noise1 = self.get_noise(self.gt.shape[2],(self.gt.shape[0],self.gt.shape[1])).to(self.args.device).float()
        self.noise2 = self.get_noise(self.gt.shape[2],(self.gt.shape[0],self.gt.shape[1])).to(self.args.device).float()

        self.net=double_U_net_skip(Out_fhsi,Out_fmsi,self.args)
        
        
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch +  1 - self.args.niter3_dip) / float(self.args.niter_decay3_dip + 1)
            return lr_l
        
        self.optimizer=optim.Adam(self.net.parameters(), lr=self.args.lr_stage3_dip)
        self.scheduler=lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        
    def get_noise(self,input_depth, spatial_size, method='2D',noise_type='u', var=1./10):
            """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
            initialized in a specific way.
            Args:
                input_depth: number of channels in the tensor
                method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
                spatial_size: spatial size of the tensor to initialize
                noise_type: 'u' for uniform; 'n' for normal
                var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
            """
            def fill_noise(x, noise_type):
                """Fills tensor `x` with noise of type `noise_type`."""
                if noise_type == 'u':
                    x.uniform_()
                elif noise_type == 'n':
                    x.normal_() 
                else:
                    assert False
            
            if isinstance(spatial_size, int):
                spatial_size = (spatial_size, spatial_size)
                
            if method == '2D':
                shape = [1, input_depth, spatial_size[0], spatial_size[1]] 
            elif method == '3D':
                shape = [1, 1, input_depth, spatial_size[0], spatial_size[1]]
            else:
                assert False
        
            net_input = torch.zeros(shape)
            
            fill_noise(net_input, noise_type)
            net_input *= var            
    
            
            return net_input
    
    def train(self):
        flag_best_fhsi=[10,0,'data',0] #第一个是SAM，第二个是PSNR,第三个为恢复的图像,第四个是保存最佳结果对应的epoch
        flag_best_fmsi=[10,0,'data',0]
        
        
        L1Loss = nn.L1Loss(reduction='mean')
        
        for epoch in range(1, self.args.niter3_dip + self.args.niter_decay3_dip + 1):
        
            
            self.optimizer.zero_grad()
            
            self.hrhsi_fhsi,self.hrhsi_fmsi=self.net(self.Out_fhsi,self.Out_fmsi)
            
          
            
            ''' generate hr_msi_est '''
            #print(self.hrhsi_est.shape)
            self.hr_msi_hrhsi_fhsi = self.srf_down(self.hrhsi_fhsi,self.srf_est)
            self.hr_msi_hrhsi_fmsi = self.srf_down(self.hrhsi_fmsi,self.srf_est)
            
            #print("self.hr_msi_from_hrhsi shape:{}".format(self.hr_msi_from_hrhsi.shape))

            ''' generate lr_hsi_est '''
            self.lr_hsi_hrhsi_fhsi = self.psf_down(self.hrhsi_fhsi, self.psf_est, self.args.scale_factor)
            self.lr_hsi_hrhsi_fmsi = self.psf_down(self.hrhsi_fmsi, self.psf_est, self.args.scale_factor)
            
            #print("self.lr_hsi_from_hrhsi shape:{}".format(self.lr_hsi_from_hrhsi.shape))
            
            loss_fhsi= L1Loss(self.hr_msi,self.hr_msi_hrhsi_fhsi) + L1Loss(self.lr_hsi,self.lr_hsi_hrhsi_fhsi)
            loss_fmsi= L1Loss(self.hr_msi,self.hr_msi_hrhsi_fmsi) + L1Loss(self.lr_hsi,self.lr_hsi_hrhsi_fmsi)
            loss=loss_fhsi+loss_fmsi
            loss.backward()
            
            
            self.optimizer.step()
                
            
            self.scheduler.step()
            
            
            if epoch % 50 ==0:

                with torch.no_grad():
                    
                    #print("____________________________________________")
                    print('epoch:{} lr:{}'.format(epoch,self.optimizer.param_groups[0]['lr']))
                    print('************')
                    
                    
                    #转为W H C的numpy 方便计算指标
                    #hrmsi
                    hr_msi_numpy=self.hr_msi.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    hr_msi_estfhsi_numpy=self.hr_msi_hrhsi_fhsi.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    hr_msi_estfmsi_numpy=self.hr_msi_hrhsi_fmsi.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    
                    #lrhsi
                    lr_hsi_numpy=self.lr_hsi.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    lr_hsi_estfhsi_numpy=self.lr_hsi_hrhsi_fhsi.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    lr_hsi_estfmsi_numpy=self.lr_hsi_hrhsi_fmsi.data.cpu().detach().numpy()[0].transpose(1,2,0)

                    #gt
                    hrhsi_est_numpy_fhsi=self.hrhsi_fhsi.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    hrhsi_est_numpy_fmsi=self.hrhsi_fmsi.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    #self.gt
                
                    ''' for fhsi'''
                    
                    #学习到的lrhsi与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(lr_hsi_numpy,lr_hsi_estfhsi_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( lr_hsi_numpy - lr_hsi_estfhsi_numpy ))
                    information1="生成lrhsi_fhsi与目标lrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information1) #监控训练过程
                    print('************')
                
                    #学习到的hrmsi与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(hr_msi_numpy,hr_msi_estfhsi_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( hr_msi_numpy - hr_msi_estfhsi_numpy ))
                    information2="生成hrmsi_fhsi与目标hrmsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information2) #监控训练过程
                    print('************')
                    
                    
                    #学习到的gt与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(self.gt,hrhsi_est_numpy_fhsi, self.args.scale_factor)
                    L1=np.mean( np.abs( self.gt - hrhsi_est_numpy_fhsi ))
                    information3="生成hrhsi_est_fhsi与目标hrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information3) #监控训练过程
                    print('************')
                   
                    file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
                    with open(file_name, 'a') as opt_file:
                        
                        opt_file.write('——————————————————epoch:{}——————————————————'.format(epoch))
                        opt_file.write('\n')
                        opt_file.write(information1)
                        opt_file.write('\n')
                        opt_file.write(information2)
                        opt_file.write('\n')
                        opt_file.write(information3)
                        opt_file.write('\n')
                        
                    
                    if sam < flag_best_fhsi[0] and psnr > flag_best_fhsi[1]:         
                  
                        flag_best_fhsi[0]=sam
                        flag_best_fhsi[1]=psnr
                        flag_best_fhsi[2]=self.hrhsi_fhsi #保存四维tensor
                        flag_best_fhsi[3]=epoch
                        
                        information_a=information1
                        information_b=information2
                        information_c=information3
                                               
                    ''' for fhsi'''
                    
                    print('--------------------------------')
                    
                    ''' for fmsi'''
                    #学习到的lrhsi与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(lr_hsi_numpy,lr_hsi_estfmsi_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( lr_hsi_numpy - lr_hsi_estfmsi_numpy ))
                    information1="生成lrhsi_fmsi与目标lrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information1) #监控训练过程
                    print('************')
                
                    #学习到的hrmsi与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(hr_msi_numpy,hr_msi_estfmsi_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( hr_msi_numpy - hr_msi_estfmsi_numpy ))
                    information2="生成hrmsi_fmsi与目标hrmsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information2) #监控训练过程
                    print('************')
                    
                    
                    #学习到的gt与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(self.gt,hrhsi_est_numpy_fmsi, self.args.scale_factor)
                    L1=np.mean( np.abs( self.gt - hrhsi_est_numpy_fmsi ))
                    information3="生成hrhsi_est_fmsi与目标hrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information3) #监控训练过程
                    print('************')
                   
                    print('————————————————————————————————')
                    
                    file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
                    with open(file_name, 'a') as opt_file:
                        
                        #opt_file.write('epoch:{}'.format(epoch))
                        opt_file.write('-------------------------------')
                        opt_file.write('\n')
                        opt_file.write(information1)
                        opt_file.write('\n')
                        opt_file.write(information2)
                        opt_file.write('\n')
                        opt_file.write(information3)
                        opt_file.write('\n')
                        #opt_file.write('————————————————————————————————')
                    
                    if sam < flag_best_fmsi[0] and psnr > flag_best_fmsi[1]:         
                    
                        flag_best_fmsi[0]=sam
                        flag_best_fmsi[1]=psnr
                        flag_best_fmsi[2]=self.hrhsi_fmsi #保存四维tensor
                        flag_best_fmsi[3]=epoch
                        
                        information_d=information1
                        information_e=information2
                        information_f=information3                   
                    ''' for fmsi'''
                        
        
        #保存最好的结果
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fhsi_S3.mat'), {'Out':flag_best_fhsi[2].data.cpu().numpy()[0].transpose(1,2,0)})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fmsi_S3.mat'), {'Out':flag_best_fmsi[2].data.cpu().numpy()[0].transpose(1,2,0)})

        
        
        #保存精度
        file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
        with open(file_name, 'a') as opt_file:
            
            opt_file.write('————————————最终结果————————————')
            opt_file.write('\n')
            opt_file.write('epoch_fhsi_best:{}'.format(flag_best_fhsi[3]))
            opt_file.write('\n')
            opt_file.write(information_a)
            opt_file.write('\n')
            opt_file.write(information_b)
            opt_file.write('\n')
            opt_file.write(information_c)
            opt_file.write('\n')
            opt_file.write('epoch_fmsi_best:{}'.format(flag_best_fmsi[3]))
            opt_file.write('\n')
            opt_file.write(information_d)
            opt_file.write('\n')
            opt_file.write(information_e)
            opt_file.write('\n')
            opt_file.write(information_f)
        
            
        return flag_best_fhsi[2] ,flag_best_fmsi[2]
    

        
        
if __name__ == "__main__":
    
    pass
    