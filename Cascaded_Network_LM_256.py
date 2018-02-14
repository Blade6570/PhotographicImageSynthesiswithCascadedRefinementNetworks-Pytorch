#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:45:23 2017

@author: soumya
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os, scipy.io
import torch.utils.model_zoo as model_zoo
import helper
from skimage import io
from random import shuffle
import scipy.misc


plt.ion()   # interactive mode


__all__ = ['VGG19', 'vgg19']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-12, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):

        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

class cascaded_model(nn.Module):
    
    def __init__(self, D_m):
        super(cascaded_model, self).__init__()
        self.conv1=nn.Conv2d(20, D_m[1], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv1.weight, gain=1)

        nn.init.constant(self.conv1.bias, 0)
        self.lay1=LayerNorm(D_m[1], eps=1e-12, affine=True)
        
        self.relu1=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv11=nn.Conv2d(D_m[1], D_m[1], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv11.weight, gain=1)

        nn.init.constant(self.conv11.bias, 0)
        self.lay11=LayerNorm(D_m[1], eps=1e-12, affine=True)
        
        self.relu11=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        #Layer2
        
        self.conv2=nn.Conv2d(D_m[1]+20, D_m[2], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv2.weight, gain=1)
#        nn.init.constant(self.conv2.weight, 1)
        nn.init.constant(self.conv2.bias, 0)
        self.lay2=LayerNorm(D_m[2], eps=1e-12, affine=True)
#        self.lay2=nn.BatchNorm2d(D_m[2])
        self.relu2=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv22=nn.Conv2d(D_m[2], D_m[2], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv22.weight, gain=1)
#        nn.init.constant(self.conv22.weight, 1)
        nn.init.constant(self.conv22.bias, 0)
        self.lay22=LayerNorm(D_m[2], eps=1e-12, affine=True)
#        self.lay2=nn.BatchNorm2d(D_m[2])
        self.relu22=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        
        #layer 3
        
        self.conv3=nn.Conv2d(D_m[2]+20, D_m[3], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv3.weight, gain=1)
#        nn.init.constant(self.conv3.weight,1)
        nn.init.constant(self.conv3.bias, 0)
        self.lay3=LayerNorm(D_m[3], eps=1e-12, affine=True)
#        self.lay3=nn.BatchNorm2d(D_m[3])
        self.relu3=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv33=nn.Conv2d(D_m[3], D_m[3], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv33.weight,gain=1)
        nn.init.constant(self.conv33.bias, 0)
        self.lay33=LayerNorm(D_m[3], eps=1e-12, affine=True)
#        self.lay3=nn.BatchNorm2d(D_m[3])
        self.relu33=nn.LeakyReLU(negative_slope=0.2,inplace=True)
               
        #layer4
                
        self.conv4=nn.Conv2d(D_m[3]+20, D_m[4], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv4.weight,gain=1)
        nn.init.constant(self.conv4.bias, 0)
        self.lay4=LayerNorm(D_m[4], eps=1e-12, affine=True)
#        self.lay4=nn.BatchNorm2d(D_m[4])
        self.relu4=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv44=nn.Conv2d(D_m[4], D_m[4], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv44.weight,gain=1)
        nn.init.constant(self.conv44.bias, 0)
        self.lay44=LayerNorm(D_m[4], eps=1e-12, affine=True)
#        self.lay4=nn.BatchNorm2d(D_m[4])
        self.relu44=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        #layers5 
        
        self.conv5=nn.Conv2d(D_m[4]+20, D_m[5], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv5.weight, gain=1)
        nn.init.constant(self.conv5.bias, 0)
        self.lay5=LayerNorm(D_m[5], eps=1e-12, affine=True)
#        self.lay5=nn.BatchNorm2d(D_m[5])
        self.relu5=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv55=nn.Conv2d(D_m[5], D_m[5], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv55.weight, gain=1)
        nn.init.constant(self.conv55.bias, 0)
        self.lay55=LayerNorm(D_m[5], eps=1e-12, affine=True)
#        self.lay5=nn.BatchNorm2d(D_m[5])
        self.relu55=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        #layer 6
        
        self.conv6=nn.Conv2d(D_m[5]+20, D_m[6], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv6.weight, gain=1)
        nn.init.constant(self.conv6.bias, 0)
        self.lay6=LayerNorm(D_m[6], eps=1e-12, affine=True)
#        self.lay6=nn.BatchNorm2d(D_m[6])
        self.relu6=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv66=nn.Conv2d(D_m[6], D_m[6], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv66.weight, gain=1)
        nn.init.constant(self.conv66.bias, 0)
        self.lay66=LayerNorm(D_m[6], eps=1e-12, affine=True)
#        self.lay6=nn.BatchNorm2d(D_m[6])
        self.relu66=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        #layer7
        self.conv7=nn.Conv2d(D_m[6]+20, D_m[6], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv7.weight, gain=1)
        nn.init.constant(self.conv7.bias, 0)
        self.lay7=LayerNorm(D_m[6], eps=1e-12, affine=True)
#        self.lay6=nn.BatchNorm2d(D_m[6])
        self.relu7=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv77=nn.Conv2d(D_m[6], D_m[6], kernel_size=3, stride=1, padding=1,bias=True)
        nn.init.xavier_uniform(self.conv77.weight, gain=1)
        nn.init.constant(self.conv77.bias, 0)
        self.lay77=LayerNorm(D_m[6], eps=1e-12, affine=True)
#        self.lay6=nn.BatchNorm2d(D_m[6])
        self.relu77=nn.LeakyReLU(negative_slope=0.2,inplace=True)
        
        self.conv8=nn.Conv2d(D_m[6], 27, kernel_size=1, stride=1, padding=0,bias=True)
        nn.init.xavier_uniform(self.conv8.weight, gain=1)
        nn.init.constant(self.conv8.bias, 0)
    def forward(self, D, label):
        
        out1= self.conv1(D[1])
        L1=self.lay1(out1)
        out2= self.relu1(L1)
        
        out11= self.conv11(out2)
        L11=self.lay11(out11)
        out22= self.relu11(L11)
        
        m = nn.Upsample(size=(D[1].size(3),D[1].size(3)*2), mode='bilinear')        
        
        img1 = torch.cat((m(out22), D[2]),1)        
        
        out3= self.conv2(img1)
        L2=self.lay2(out3)
        out4= self.relu2(L2)
        
        out33= self.conv22(out4)
        L22=self.lay22(out33)
        out44= self.relu22(L22)
        
        m = nn.Upsample(size=(D[2].size(3),D[2].size(3)*2), mode='bilinear')
        
        img2 = torch.cat((m(out44), D[3]),1)
        
        out5= self.conv3(img2)
        L3=self.lay3(out5)
        out6= self.relu3(L3)
        
        out55= self.conv33(out6)
        L33=self.lay33(out55)
        out66= self.relu33(L33)
        
        m = nn.Upsample(size=(D[3].size(3),D[3].size(3)*2),mode='bilinear')
        
        img3 = torch.cat((m(out66), D[4]),1)
        
        out7= self.conv4(img3)
        L4=self.lay4(out7)
        out8= self.relu4(L4)
        
        out77= self.conv44(out8)
        L44=self.lay44(out77)
        out88= self.relu44(L44)

        m = nn.Upsample(size=(D[4].size(3),D[4].size(3)*2),mode='bilinear')
        
        img4 = torch.cat((m(out88), D[5]),1)        
        
        out9= self.conv5(img4)
        L5=self.lay5(out9)
        out10= self.relu5(L5)
        
        out99= self.conv55(out10)
        L55=self.lay55(out99)
        out110= self.relu55(L55)
#        L5=self.lay5(out10)
        
        m = nn.Upsample(size=(D[5].size(3),D[5].size(3)*2),mode='bilinear')
        
        img5 = torch.cat((m(out110), D[6]),1)
               
        out11= self.conv6(img5)
        L6=self.lay6(out11)
        out12= self.relu6(L6)
        
        out111= self.conv66(out12)
        L66=self.lay66(out111)
        out112= self.relu66(L66)
        
        m = nn.Upsample(size=(D[6].size(3),D[6].size(3)*2),mode='bilinear')
        
        img6 = torch.cat((m(out112), label),1)       
         
        out13= self.conv7(img6)
        L7=self.lay7(out13)
        out14= self.relu7(L7)
        
        out113= self.conv77(out14)
        L77=self.lay77(out113)
        out114= self.relu77(L77)
        
        out15= self.conv8(out114)
        
        out15=(out15+1.0)/2.0*255.0
        
        out16,out17,out18=torch.chunk(out15.permute(1,0,2,3),3,0)
        out=torch.cat((out16,out17,out18),1)

        return out
        
      
        
class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        self.conv1=nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1=nn.ReLU(inplace=True)
            
        self.conv2=nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2=nn.ReLU(inplace=True)
        self.max1=nn.AvgPool2d(kernel_size=2, stride=2)
            
        self.conv3=nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.relu3=nn.ReLU(inplace=True)
            
        self.conv4=nn.Conv2d(128, 128,  kernel_size=3, padding=1, bias=True)
        self.relu4=nn.ReLU(inplace=True)
        self.max2=nn.AvgPool2d(kernel_size=2, stride=2)
            
        self.conv5=nn.Conv2d(128, 256,  kernel_size=3, padding=1, bias=True)
        self.relu5=nn.ReLU(inplace=True)
            
        self.conv6=nn.Conv2d(256, 256,  kernel_size=3, padding=1, bias=True)
        self.relu6=nn.ReLU(inplace=True)
            
        self.conv7=nn.Conv2d(256, 256,  kernel_size=3, padding=1, bias=True)
        self.relu7=nn.ReLU(inplace=True)
            
        self.conv8=nn.Conv2d(256, 256,  kernel_size=3, padding=1, bias=True)
        self.relu8=nn.ReLU(inplace=True)
        self.max3=nn.AvgPool2d(kernel_size=2, stride=2)
            
        self.conv9=nn.Conv2d(256, 512,  kernel_size=3, padding=1, bias=True)
        self.relu9=nn.ReLU(inplace=True)
            
        self.conv10=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu10=nn.ReLU(inplace=True)
            
        self.conv11=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu11=nn.ReLU(inplace=True)
            
        self.conv12=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu12=nn.ReLU(inplace=True)
        self.max4=nn.AvgPool2d(kernel_size=2, stride=2)
            
        self.conv13=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu13=nn.ReLU(inplace=True)
            
        self.conv14=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu14=nn.ReLU(inplace=True)
            
        self.conv15=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu15=nn.ReLU(inplace=True)
            
        self.conv16=nn.Conv2d(512, 512,  kernel_size=3, padding=1, bias=True)
        self.relu16=nn.ReLU(inplace=True)
        self.max5=nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        
        out1= self.conv1(x)
        out2= self.relu1(out1)
            
        out3= self.conv2(out2)
        out4=self.relu2(out3)
        out5=self.max1(out4)
            
        out6=self.conv3(out5)
        out7=self.relu3(out6)            
        out8=self.conv4(out7)
        out9=self.relu4(out8)
        out10=self.max2(out9)          
        out11=self.conv5(out10)
        out12=self.relu5(out11)           
        out13=self.conv6(out12)
        out14=self.relu6(out13)            
        out15=self.conv7(out14)
        out16=self.relu7(out15)            
        out17=self.conv8(out16)
        out18=self.relu8(out17)
        out19=self.max3(out18)           
        out20=self.conv9(out19)
        out21=self.relu9(out20)            
        out22=self.conv10(out21)
        out23=self.relu10(out22)            
        out24=self.conv11(out23)
        out25=self.relu11(out24)           
        out26=self.conv12(out25)
        out27=self.relu12(out26)
        out28=self.max4(out27)           
        out29=self.conv13(out28)
        out30=self.relu13(out29)           
        out31=self.conv14(out30)
        out32=self.relu14(out31)            
        out33=self.conv15(out32)
        out34=self.relu15(out33)            
        out35=self.conv16(out34)
        out36=self.relu16(out35)
        out37=self.max5(out36)
        return out4, out9, out14, out23, out32, out7                     #Add appropriate outputs


def vggnet(pretrained=False, model_root=None, **kwargs):
    model = VGG19(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root))
    return model

Net=vggnet(pretrained=False, model_root=None)

Net=Net.cuda()

vgg_rawnet=scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')

vgg_layers=vgg_rawnet['layers'][0]

#Weight initialization according to the pretrained VGG Very deep 19 network Network weights

layers=[0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]

att=['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13', 'conv14', 'conv15', 'conv16']

S=[64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
for L in range(16):
#    getattr(Net, att[L]).weight=nn.Parameter(torch.from_numpy(vgg_layers[layers[L]][0][0][2][0][0].reshape(S[L],-1,3,3)))
    getattr(Net, att[L]).weight=nn.Parameter(torch.from_numpy(vgg_layers[layers[L]][0][0][2][0][0]).permute(3,2,0,1).cuda())
    getattr(Net, att[L]).bias=nn.Parameter(torch.from_numpy(vgg_layers[layers[L]][0][0][2][0][1]).view(S[L]).cuda())
    
#Till Now VGG19 pretrained network is ready
    
#Cascaded Refinement Network will start from now
global D_m
global D
global count
D=[]
D_m=[]
count=0

def recursive_img(label,res): #Resulution may refers to the final image output i.e. 256x512 or 512x1024
     dim=512 if res>=128 else 1024
#    #M_low will start from 4x8 to resx2*res
     if res == 4:
         downsampled = label #torch.unsqueeze(torch.from_numpy(label).float().permute(2,0,1), dim=0)
     else:
         max1=nn.AvgPool2d(kernel_size=2, padding=0, stride=2)
         downsampled=max1(label)
         img = recursive_img(downsampled, res//2)
         
     global D
     global count
     global D_m
     D.insert(count, downsampled)
     D_m.insert(count, dim)
     count+=1
     return downsampled  
# Loss function goes here
     
def compute_error(R,F,label_images):
    E=torch.mean(torch.mean(label_images* torch.mean(torch.abs(R-F),1).unsqueeze(1),2),2)
#    E= torch.mean(torch.abs(R-F))
    return E

def loss_function(real, generator,label_images,D):
    
    aa=np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
    bb=Variable(torch.from_numpy(aa).float().permute(0,3,1,2).cuda())
    out3_r, out8_r, out13_r, out22_r, out33_r, out7r =Net(real-bb)
    out3_f, out8_f, out13_f, out22_f, out33_f, out7f =Net(generator-bb)
    
    E0=compute_error(real-bb,generator-bb,label_images)
    E1=compute_error(out3_r,out3_f,label_images)/1.6
    E2=compute_error(out8_r,out8_f,D[6])/2.3
    E3=compute_error(out13_r,out13_f,D[5])/1.8
    E4=compute_error(out22_r,out22_f,D[4])/2.8
    E5=compute_error(out33_r,out33_f,D[3])*10/0.8
    Total_loss=E0+E1+E2+E3+E4+E5
    aa=torch.min(Total_loss, 0)
    G_loss=torch.sum(aa[0])*0.999+torch.sum(torch.mean(Total_loss, 0))*0.001
    #G_loss=torch.sum(torch.min(Total_loss, 0))*0.999+torch.sum(torch.mean(Total_loss, 0))*0.001
    return G_loss
    
def training(M):      
     res=256
         
     label_dir='Label256Full'
     l=os.listdir(label_dir)
     
     for epoch in range(100):
         running_loss=0
         c_t=0
         for I in enumerate(l):
             c_t+=1
             global D_m
             global D
             global count
             D=[]
             D_m=[]
             count=0
             J=str.replace(I[1],'gtFine_color.png', 'leftImg8bit.png')
         
             label_images1=Variable(torch.unsqueeze(torch.from_numpy(helper.get_semantic_map('Label256Full/'+I[1])).float().permute(2,0,1), dim=0))#.cuda()#training label
             input_images=Variable(torch.unsqueeze(torch.from_numpy(io.imread("RGB256Full/"+J)).float(),dim=0).permute(0,3,1,2))
             label_images = torch.cat((label_images1, (1-label_images1.sum(1)).unsqueeze(1)),1)
             input_images=input_images.cuda()
             label_images=label_images.cuda()
             G_temp=recursive_img(label_images,res)
             if M==0:
                 model=cascaded_model(D_m)
                 model=model.cuda()
#                 model.load_state_dict(torch.load('mynet_updated.pth')) # if u want to resume training from a pretrained model then add the .pth file here
                 optimizer=optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
              
             optimizer.zero_grad()
             Generator=model(D, label_images)
             Loss=loss_function(input_images, Generator,label_images,D)             
                  
             Loss.backward()
             optimizer.step()
             M=1
             running_loss += Loss.data[0]
             print(epoch,c_t,Loss.data[0])
             del D
             del D_m
             del count
             del Loss, label_images,G_temp, input_images
         shuffle(l)
         epoch_loss = running_loss / 2975.0 #can replace the 2975 with c_t for generalization
         print(epoch, epoch_loss)
         if epoch % 2 == 0:
            Generator=Generator.permute(0,2,3,1)
            Generator=Generator.cpu()
            Generator=Generator.data.numpy()
            output=np.minimum(np.maximum(Generator,0.0), 255.0)
            scipy.misc.toimage(output[0,:,:,:],cmin=0,cmax=255).save("%06d_output_real.jpg"%epoch)
         #epoch_acc = running_corrects / 2975.0
     
#     return Loss
     best_model_wts = model.state_dict()
     model.load_state_dict(best_model_wts)
     
     return model

def testing(seman_in):
   label_images1=Variable(torch.unsqueeze(torch.from_numpy(helper.get_semantic_map(seman_in)).float().permute(2,0,1), dim=0))
   global D_m
   global D
   global count
   D=[]
   D_m=[]
   count=0

   label_images = torch.cat((label_images1, (1-label_images1.sum(1)).unsqueeze(1)),1)
   label_images=label_images#.cuda()
   res=256
   G_temp=recursive_img(label_images,res)    
   model=cascaded_model(D_m)
   model=model.cuda()
   model.load_state_dict(torch.load('mynet_200epoch_CRN.pth'))
   model=model.cpu().eval()
   G=model(D, label_images)
   Generator=G.permute(0,2,3,1)
   Generator=Generator
   Generator=Generator.data.numpy()
   output=np.minimum(np.maximum(Generator,0.0), 255.0)
   scipy.misc.toimage(output[2,:,:,:],cmin=0,cmax=255).save("val3.jpg")


mode='test'

if mode=='train':
   M=0
   model_ft=training(M)
   torch.save(model_ft.state_dict(),'mynet_200epoch_CRN.pth')    
else: 
    file_name='/home/soumya/Documents/cascaded_code_for_cluster/Label256Fullval/frankfurt_000000_000294_gtFine_color.png'
    testing(file_name)


                  
    
    
