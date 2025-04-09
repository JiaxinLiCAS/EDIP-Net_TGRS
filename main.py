# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug
空天信息创新研究院20-25直博生，导师高连如
"""
"""
❗❗❗❗❗❗#主文件
"""

import torch
import time
import numpy as np
import random
from model.config import args

#设置随机种子
def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
        
setup_seed(args.seed) 


'''第一阶段'''
from model.srf_psf_layer import Blind       #将退化函数看作一层的参数
blind = Blind(args)


start = time.perf_counter() # 记录开始时间
lr_msi_fhsi_est, lr_msi_fmsi_est=blind.train() #device格式的1 C H W张量
blind.get_save_result() #保存PSF SRF
psf = blind.model.psf.data.cpu().detach().numpy()[0,0,:,:] #15 x 15  numpy
srf = blind.model.srf.data.cpu().detach().numpy()[:,:,0,0].T #46 x 8   numpy
psf_gt=blind.psf_gt #15 x 15   numpy
srf_gt=blind.srf_gt  #46 x 8   numpy

end = time.perf_counter()   # 记录结束时间
elapsed_S1 = end - start        # 计算经过的时间（单位为秒）

'''第二阶段'''
from model.spectral_up import spectral_SR
    
spectral_sr=spectral_SR(args,lr_msi_fhsi_est.clone().detach(), lr_msi_fmsi_est.clone().detach(),  #lr_msi_fhsi_est, lr_msi_fmsi_est
                           blind.tensor_lr_hsi,blind.tensor_hr_msi,blind.gt) 
start = time.perf_counter() # 记录开始时间
Out_fhsi_s2,Out_fmsi_s2=spectral_sr.train() #返回的是四维tensor device上
end = time.perf_counter()   # 记录结束时间
elapsed_S2 = end - start        # 计算经过的时间（单位为秒）  

'''第三阶段''' 
''' 第三阶段耗时最多，如果你觉得代码运行过慢，可以将config.py里的--band参数调小'''

from model.dip import dip #my_dip  ,  dip 
DIP=dip(args,Out_fhsi_s2,Out_fmsi_s2,psf,srf,blind)
start = time.perf_counter() # 记录开始时间  
out_fhsi_s3,out_fmsi_s3=DIP.train() # 返回的是四维tensor device上
end = time.perf_counter()   # 记录结束时间
elapsed_S3 = end - start        # 计算经过的时间（单位为秒） 

'''第四阶段'''
from model.select import select_decision
start = time.perf_counter() # 记录开始时间  
srf_out=select_decision(out_fhsi_s3,out_fmsi_s3,blind,decide='rmse') #返回的是3维H W C numpy 这个就是最终的结果
end = time.perf_counter()   # 记录结束时间
elapsed_S4 = end - start        # 计算经过的时间（单位为秒）

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(blind.model))
print(get_parameter_number(spectral_sr.two_stream))
print(get_parameter_number(DIP.net))
print("training time S1:{},s2:{},s3:{},s4:{}".format(elapsed_S1,elapsed_S2,elapsed_S3,elapsed_S4))

print('李嘉鑫 作者微信 BatAug 空天信息创新研究院20-25直博生，导师高连如 ')

"""
❗❗❗❗❗❗李嘉鑫 
空天信息创新研究院20-25直博生，导师高连如
作者微信 BatAug
"""





