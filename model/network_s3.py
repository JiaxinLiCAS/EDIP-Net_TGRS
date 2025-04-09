# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug
空天信息创新研究院20-25直博生，导师高连如

"""
"""
❗❗❗❗❗❗#此py作用：第三阶段所需要的网络模块
"""
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np
import os
import scipy
import torch.nn.functional as fun

def init_weights(net, init_type, gain):
    print('in init_weights')
    def init_func(m):
        classname = m.__class__.__name__
        #print(classname,m,'_______')
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(height*weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net,device, init_type, init_gain,initializer):
    print('in init_net')
    net.to(device)  #gpu_ids[0] 是 gpu_ids列表里面的第一个int值
    if initializer :
        #print(2,initializer)
        init_weights(net,init_type, init_gain)
    else:
        print('Spectral_downsample with default initialize')
    return net

    




########################## double_U_net_skip ############################

def double_U_net_skip(Out_fhsi,Out_fmsi,args, init_type='kaiming', init_gain=0.02,initializer=True ):
    
    net = double_u_net_skip(Out_fhsi,Out_fmsi,args)

    return init_net(net,args.device, init_type, init_gain,initializer)

class double_u_net_skip(nn.Module):
    def __init__(self,Out_fhsi,Out_fmsi,args): 

        super().__init__()
        
        
        self.band=args.band
        #self.fusion=fusion
        self.Out_fhsi=Out_fhsi
        self.Out_fmsi=Out_fmsi
        self.scale=[
                              (  self.Out_fhsi.shape[2],self.Out_fhsi.shape[3]  ),
                              (  int(self.Out_fhsi.shape[2]/2),int(self.Out_fhsi.shape[3]/2)  ),
                              (  int(self.Out_fhsi.shape[2]/4), int(self.Out_fhsi.shape[3]/4) )
                              ]
        print(self.scale)
       
        
        
        '''for out_fhsi'''
        self.ex1=nn.Sequential(
        nn.Conv2d(self.Out_fhsi.shape[1],self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True) nn.ReLU(inplace=True) 
                                )
        
        self.ex2=nn.Sequential(
        nn.Conv2d(self.band,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.ex3=nn.Sequential(
        nn.Conv2d(self.band,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.ex4=nn.Sequential(
        nn.Conv2d(self.band+2,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.ex5=nn.Sequential(
        nn.Conv2d(self.band+2,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        #nn.Sigmoid()  #nn.LeakyReLU(0.2, inplace=True)
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(self.band,self.Out_fhsi.shape[1],kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.Sigmoid()
                                )
        
        self.skip1=nn.Sequential(
        nn.Conv2d(self.band,2,kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.BatchNorm2d(2),
        nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.skip2=nn.Sequential(
        nn.Conv2d(self.band,2,kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.BatchNorm2d(2),
        nn.LeakyReLU(0.2, inplace=True)
                                )
        '''for out_fhsi'''
        
        '''for out_fmsi'''
        
        self.ex6=nn.Sequential(
        nn.Conv2d(self.Out_fmsi.shape[1],self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True) nn.ReLU(inplace=True) 
                                )
        
        self.ex7=nn.Sequential(
        nn.Conv2d(self.band,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.ex8=nn.Sequential(
        nn.Conv2d(self.band,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.ex9=nn.Sequential(
        nn.Conv2d(self.band+2,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True) #nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.ex10=nn.Sequential(
        nn.Conv2d(self.band+2,self.band,kernel_size=(5,5),stride=1,padding=(2,2)) ,
        #nn.Sigmoid()  #nn.LeakyReLU(0.2, inplace=True)
        nn.BatchNorm2d(self.band),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(self.band,self.Out_fmsi.shape[1],kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.Sigmoid()
                                )
        
        self.skip3=nn.Sequential(
        nn.Conv2d(self.band,2,kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.BatchNorm2d(2),
        nn.LeakyReLU(0.2, inplace=True)
                                )
        
        self.skip4=nn.Sequential(
        nn.Conv2d(self.band,2,kernel_size=(1,1),stride=1,padding=(0,0)) ,
        nn.BatchNorm2d(2),
        nn.LeakyReLU(0.2, inplace=True)
                                )
        '''for out_fmsi'''
        
        
    def forward(self,Out_fhsi,Out_fmsi):
        
        '''for out_fhsi'''
        #nn.Upsample(scale_factor=2, mode=upsample_mode[i])
        #down
        x1=self.ex1(Out_fhsi)
        #x2=fun.interpolate(x1,self.scale[1],mode='bilinear') 
        x2=nn.AdaptiveAvgPool2d(self.scale[1])(x1)
        x3=self.ex2(x2)
        #x4=fun.interpolate(x3,self.scale[2],mode='bilinear')
        x4=nn.AdaptiveAvgPool2d(self.scale[2])(x3)

        x5=self.ex3(x4)
        
        #up
        up=nn.Upsample(self.scale[1], mode='bilinear')
        s1=self.skip1(x3)
        #x6=fun.interpolate(x5,self.scale[1],mode='bilinear')
        x6=up(x5)
        x7=self.ex4(torch.cat([s1,x6],dim=1))
        
        up=nn.Upsample(self.scale[0], mode='bilinear')
        s2=self.skip2(x1)
        #x8=fun.interpolate(x7,self.scale[0],mode='bilinear')
        x8=up(x7)
        out_fhsi=self.ex5(torch.cat([s2,x8],dim=1))
        '''for out_fhsi'''
        
        
        '''for out_fmsi'''
        #nn.Upsample(scale_factor=2, mode=upsample_mode[i])
        #down
        x9=self.ex6(Out_fmsi)
        #x2=fun.interpolate(x1,self.scale[1],mode='bilinear') 
        x10=nn.AdaptiveAvgPool2d(self.scale[1])(x9)
        x11=self.ex7(x10)
        #x4=fun.interpolate(x3,self.scale[2],mode='bilinear')
        x12=nn.AdaptiveAvgPool2d(self.scale[2])(x11)

        x13=self.ex8(x12)
        
        #up
        up=nn.Upsample(self.scale[1], mode='bilinear')
        s3=self.skip3(x11)
        #x6=fun.interpolate(x5,self.scale[1],mode='bilinear')
        x14=up(x13)
        x15=self.ex9(torch.cat([s3,x14],dim=1))
        
        up=nn.Upsample(self.scale[0], mode='bilinear')
        s4=self.skip4(x9)
        #x8=fun.interpolate(x7,self.scale[0],mode='bilinear')
        x16=up(x15)
        out_fmsi=self.ex10(torch.cat([s4,x16],dim=1))
        '''for out_fmsi'''
        
        
        return out_fhsi ,out_fmsi


########################## double_U_net_skip############################



if __name__ == "__main__":
    pass

    