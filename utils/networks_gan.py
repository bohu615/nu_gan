import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
from torch.autograd import Variable
from imgaug import augmenters as iaa

from evaluation import compute_purity_entropy, get_f_score

np.random.seed(42)

#create_resblocks
class avgpool(nn.Module):
    def __init__(self, up_size=0):
        super(avgpool, self).__init__()
        
    def forward(self, x):
        out_man = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4
        return out_man
    
class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, resample=None, up_size=0):
        super(ResidualBlock, self).__init__()
        if resample == 'up':
            self.bn1 = nn.BatchNorm2d(in_dim)
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.upsample = torch.nn.Upsample(up_size,2)
            self.upsample_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            self.bn2 = nn.BatchNorm2d(out_dim)
            
        elif resample == 'down':
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            self.pool = avgpool()
            self.pool_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
        
        elif resample == None:
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            
        self.resample = resample

    def forward(self, x):
        
        if self.resample == None:
            shortcut = x
            output = x
            
            output = nn.functional.relu(output)
            output = self.conv1(output)
            output = nn.functional.relu(output)
            output = self.conv2(output)
            
        elif self.resample == 'up':
            shortcut = x
            output = x
            
            shortcut = self.upsample(shortcut) #upsampleconv
            shortcut = self.upsample_conv(shortcut)
            
            output = self.bn1(output)
            output = nn.functional.relu(output)
            output = self.conv1(output)

            output = self.bn2(output)
            output = nn.functional.relu(output)
            output = self.upsample(output) #upsampleconv
            output = self.conv2(output)
                        
        elif self.resample == 'down':
            shortcut = x
            output = x
            
            shortcut = self.pool_conv(shortcut) #convmeanpool
            shortcut = self.pool(shortcut)
            
            output = nn.functional.relu(output)
            output = self.conv1(output)
            
            output = nn.functional.relu(output)
            output = self.conv2(output)    #convmeanpool
            output = self.pool(output)
            
        return output+shortcut

class ResidualBlock_thefirstone(nn.Module):

    def __init__(self, in_dim, out_dim, resample=None, up_size=0):
        super(ResidualBlock_thefirstone, self).__init__()
        
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
        self.pool = avgpool()
        self.pool_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
        
    def forward(self, x):
        
        shortcut = x
        output = x
        
        shortcut = self.pool(shortcut) #meanpoolconv
        shortcut = self.pool_conv(shortcut)

        output = self.conv1(output)
        output = nn.functional.relu(output)
        output = self.conv2(output) #convmeanpool
        output = self.pool(output)
            
        return output+shortcut

#create_gan_architecture
class generator(nn.Module):

    def __init__(self, rand=128):
        super(generator, self).__init__()
        self.rand = rand
        self.linear = nn.Linear(rand  ,2048, bias=True)
        self.layer_up_1 = ResidualBlock(128, 128, 'up', up_size=8)
        self.layer_up_2 = ResidualBlock(128, 128, 'up', up_size=16)
        self.layer_up_3 = ResidualBlock(128, 128, 'up', up_size=32)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv_last = nn.Conv2d(128, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        x = x.view(-1,self.rand)
        x = self.linear(x)
        x = x.view(-1,128,4,4)
        x = self.layer_up_1(x)
        x = self.layer_up_2(x)
        x = self.layer_up_3(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv_last(x)
        x = nn.functional.tanh(x)
        return x

class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.layer_down_1 = ResidualBlock_thefirstone(3, 128)
        self.layer_down_2 = ResidualBlock(128, 128, 'down')
        self.layer_none_1 = ResidualBlock(128, 128, None)
        self.layer_none_2 = ResidualBlock(128, 128, None)
        
    def forward(self, x):
        x = self.layer_down_1(x)
        x = self.layer_down_2(x)
        x = self.layer_none_1(x)
        x = self.layer_none_2(x)
        x = nn.functional.relu(x)
        x = x.mean(2).mean(2)
        return x
    
class _netD_D(nn.Module):
    def __init__(self):
        super(_netD_D, self).__init__()
        self.linear = nn.Linear(128,1, bias=True)
        
    def forward(self, x):
        x = x.view(-1, 128)
        x = self.linear(x)
        return x.view(-1,1,1,1)
    
class _netD_Q(nn.Module):
    def __init__(self, nd = 10):
        super(_netD_Q, self).__init__()
        self.linear = nn.Linear(128,nd, bias=True)
        self.softmax = nn.LogSoftmax()
        self.nd = nd

    def forward(self, x):
        x = x.view(-1, 128)
        x = self.linear(x)
        x = self.softmax(x)
        return x.view(-1,self.nd,1,1)

def uniform(stdev, size):
    return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

def initialize_conv(m,he_init=True):
    fan_in = m.in_channels * m.kernel_size[0]**2
    fan_out = m.out_channels * m.kernel_size[0]**2 / (m.stride[0]**2)

    if m.kernel_size[0]==3:
        filters_stdev = np.sqrt(4./(fan_in+fan_out))
    # Normalized init (Glorot & Bengio)
    else: 
        filters_stdev = np.sqrt(2./(fan_in+fan_out))
        
    filter_values = uniform(
                    filters_stdev,
                    (m.kernel_size[0], m.kernel_size[0], m.in_channels, m.out_channels)
                )
    
    return filter_values

def initialize_linear(m):
    weight_values = uniform(
                np.sqrt(2./(m.in_features+m.out_features)),
                (m.in_features, m.out_features)
            )
    return weight_values

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight = torch.from_numpy(initialize_conv(m))
        m.weight.data.copy_(weight,broadcast=False)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_values = torch.from_numpy(initialize_linear(m))
        m.weight.data.copy_(weight_values,broadcast=False)
        m.bias.data.fill_(0)
        
#sample from categorical distribution
def sample_c(batchsize=32, dis_category=5):
    rand_c = np.zeros((batchsize,dis_category),dtype='float32')
    for i in range(0,batchsize):
        rand = np.random.multinomial(1, dis_category*[1/float(dis_category)], size=1)
        rand_c[i] = rand

    label_c = np.argmax(rand_c,axis=1)
    label_c = torch.LongTensor(label_c.astype('int'))
    rand_c = torch.from_numpy(rand_c.astype('float32'))
    return rand_c,label_c
        
def fix_noise(dis=1, rand=128, dis_category=5, row=10):
    
    fixed_z = np.random.randn(row, rand).repeat(dis_category,axis=0)
    changing_dis = np.zeros((row*dis_category,dis_category),dtype = np.float32)
    list = [n for n in range(0,dis_category)]*row
    for i in range(0,row*dis_category):
        changing_dis[i,list[i]] = 1
    map1 = np.concatenate((changing_dis,fixed_z),axis=1)
    lst = [map1.astype(np.float32)]
    return lst[0].reshape(row*dis_category,rand+dis*dis_category,1,1)

#gradient penalty
def calc_gradient_penalty(netD_D, netD, real_data, fake_data,lamda,batch_size):
    alpha = torch.rand(batch_size,1,1,1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD_D(netD(interpolates))#.view(batch_size,-1)

    gradients, = autograd.grad(outputs=disc_interpolates.sum(), inputs=interpolates,
                              create_graph=True)
        
    gradient_penalty = ((gradients.view(batch_size,-1).norm(2, dim=1) - 1) ** 2).mean()* lamda
    return gradient_penalty

def get_matrix(netD, netD_Q, cluster_loader, label, dis_category):
    predict = []
    data_iter = iter(cluster_loader)
    for iteration in data_iter:
        img, img_label = iteration
        predict_label = netD_Q(netD(Variable(img.cuda(),volatile=True)))
        predict.append(predict_label.data.cpu().numpy())    
    predict = np.concatenate(predict)
    predict_label = []
    for index in range(0, predict.shape[0]):
        predict_label.append(np.argmax(predict[index]))
    coherent_array = np.zeros((np.max(label)+1,dis_category), dtype=float)
    for index in range(0, len(predict)):
        coherent_array[label[index],predict_label[index]] +=1
    return coherent_array

#normalization and data augmentation
def normalized(array):
    X = np.asarray([x.transpose((2,0,1)) for x in array])
    X = X.astype(np.float32)/(255.0/2) - 1.0
    return X

def shuffle(array):
    p = np.random.permutation(X.shape[0])
    X = X[p]
    return X

def rotation(array):
    result_list = [array] 
    seq = iaa.Sequential([
        iaa.Flipud(1), 
    ])
    seq2= iaa.Sequential([
        iaa.Fliplr(1), 
    ])
    seq3 = iaa.Sequential([
        iaa.Flipud(1), 
        iaa.Fliplr(1), 
    ])

    result_list.append(seq.augment_images(array))
    result_list.append(seq2.augment_images(array))
    result_list.append(seq3.augment_images(array))
    result = np.concatenate(result_list,axis=0)
    return result

def create_loader(array, shuffle=False, batchsize=1):
    label = torch.LongTensor(np.zeros((array.shape[0]),dtype=int))
    data = torch.FloatTensor(array)
    dataset = torch.utils.data.TensorDataset(data,label)
    loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batchsize, num_workers=2)
    return loader

def train(cell_array, test_array, test_label, batchsize=32, rand=32, dis=1, dis_category=5):

    train = rotation(cell_array)
    train = normalized(train)
    train_loader = create_loader(train, shuffle=True, batchsize=batchsize)
    test = normalized(test_array)
    test_loader = create_loader(test, shuffle=False, batchsize=1)
    test_label = test_label
    
    netD, netG, netD_D, netD_Q = discriminator(), generator(rand = rand+dis*dis_category), _netD_D(), _netD_Q(dis_category)
    print(netD_D)
    print(netD_Q)
    netG.apply(weights_init)
    netD.apply(weights_init)
    netD_Q.apply(weights_init)
    netD_D.apply(weights_init)
    netD, netG, netD_D, netD_Q = netD.cuda(), netG.cuda(), netD_D.cuda(), netD_Q.cuda()

    def zero_grad():
        netD.zero_grad()
        netD_Q.zero_grad()
        netD_D.zero_grad()
        netG.zero_grad()

    ld = 1e-4
    lg = 1e-4
    lq = 1e-4

    optimizerD = optim.Adam([
                {'params': netD.parameters(), 'lr': ld},
                {'params': netD_D.parameters(), 'lr': ld},
            ], betas=(0.5, 0.9))

    optimizerG = optim.Adam([
                {'params': netG.parameters(), 'lr': lg},
            ], betas=(0.5, 0.9))

    optimizerQ = optim.Adam([
                    {'params': netG.parameters()},
                    {'params': netD.parameters()},
                    {'params': netD_Q.parameters()},
                ], lq, betas=(0.5, 0.9))

    optimizerQ_G = optim.Adam([
                    {'params': netG.parameters()},            
                ], lg, betas=(0.5, 0.9))

    input = torch.FloatTensor(batchsize, 3, 32, 32)
    noise = torch.FloatTensor(batchsize, rand+10*dis,1 ,1 )

    fixed_noise = torch.FloatTensor(np.random.multinomial(batchsize, 10*[0.1], size=1))
    c = torch.randn(batchsize, 10)
    z = torch.randn(batchsize, rand)

    label = torch.FloatTensor(1)

    real_label = 1
    fake_label = 0

    criterion = nn.BCELoss()
    criterion_logli = nn.NLLLoss()
    criterion_mse = nn.MSELoss()

    criterion, criterion_logli, criterion_mse = criterion.cuda(), criterion_logli.cuda(), criterion_mse.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    z, c = z.cuda(), c.cuda()

    gen_iterations = 0
    lamda = 10
    discrete_lamda = 1
    end = time.time()
    
    one = torch.FloatTensor([1])
    mone = one * -1
    one = one.cuda()
    mone = mone.cuda()
    fixed_noise = torch.from_numpy(fix_noise(dis_category=dis_category,rand=rand)).cuda()

    for epoch in range(100000):

        dataiter = iter(train_loader)
        i = 0

        while i < len(train_loader):

            for p in netD.parameters(): 
                p.requires_grad = True 
            for p in netD_D.parameters(): 
                p.requires_grad = True 

            for iter_d in range(0,5):
                if i >=len(train_loader):
                    continue

                zero_grad()
                image_, _ = dataiter.next()
                _batchsize = image_.size(0)
                image_ = image_.cuda()
                i +=1
                input.resize_as_(image_).copy_(image_)
                inputv = Variable(input)

                #train with real
                errD_real = netD_D(netD(inputv)).mean()
                errD_real.backward(mone)

                # train with fake
                rand_c,label_c = sample_c(_batchsize,dis_category=dis_category)
                rand_c = rand_c.cuda()
                c.resize_as_(rand_c).copy_(rand_c)
                z.resize_(_batchsize, rand, 1, 1).normal_(0, 1)
                noise = torch.cat([c,z],1)
                noise_resize = noise.view(_batchsize,rand+dis_category*dis,1,1)
                noisev = Variable(noise_resize, volatile = True)
                fake = Variable(netG(noisev).data)
                inputv = fake
                errD_fake = netD_D(netD(inputv)).mean()
                errD_fake.backward(one)

                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty(netD_D,netD, input, fake.data,lamda,_batchsize)
                gradient_penalty.backward()

                D_cost = -errD_real + errD_fake + gradient_penalty

                optimizerD.step()

            # update G  
            for p in netD.parameters(): 
                p.requires_grad = False 
            for p in netD_D.parameters(): 
                p.requires_grad = False 

            zero_grad()
            rand_c,label_c = sample_c(batchsize,dis_category=dis_category)
            rand_c = rand_c.cuda()
            c.resize_as_(rand_c).copy_(rand_c)
            z.resize_(batchsize, rand, 1, 1).normal_(0, 1)
            noise = torch.cat([c,z],1)
            noise_resize = noise.view(batchsize,rand+dis_category*dis,1,1)
            noisev = Variable(noise_resize)
            fake = netG(noisev)
            errG = netD_D(netD(fake)).mean()
            errG.backward(mone)
            optimizerG.step()

            for p in netD.parameters(): 
                p.requires_grad = True 
            for p in netD_D.parameters(): 
                p.requires_grad = True 

            zero_grad()
            inputv = Variable(noise_resize)
            Q_c_given_x = netD_Q(netD(netG(inputv))).view(batchsize, dis_category)
            crossent_loss = criterion_logli(Q_c_given_x ,Variable(label_c.cuda()))
            mi_loss = discrete_lamda*crossent_loss
            mi_loss.backward()

            optimizerQ.step()
            gen_iterations += 1

            if gen_iterations % 1 == 0 :

                batch_time = time.time() - end
                end = time.time()

                with open("./log","a") as f:
                    f.write('batch_time:{0}, gen_iterations:{1}, D_cost:{2}, mi_loss:{3}'.format(batch_time, gen_iterations , -D_cost.data[0] , mi_loss.data[0]) + '\n')


            if gen_iterations % 100 == 0 :

                print ('{0} {1} {2} {3}'.format(batch_time, gen_iterations , -D_cost.data[0] , mi_loss.data[0]))
                G_sample = netG(Variable(fixed_noise, volatile = True))
                vutils.save_image(G_sample.data, './fake_image/fake_cell_{0}.png'.format(gen_iterations),nrow=5,normalize=True)

                coherent_array = get_matrix(netD, netD_Q, test_loader, test_array, dis_category)
                entropy, purity = compute_purity_entropy(coherent_array)
                f_score = get_f_score(coherent_array)
                print('purity:', purity, 'entropy:', entropy, 'f_score', f_score)

                with open("./log_eval","a") as f:
                    f.write('{0} {1} {2} {3}'.format(gen_iterations, purity , entropy , f_score) + '\n')

                if (purity>0.70):
                    torch.save(netD.state_dict(), './model/32_linear_cell5n_netD_'+str(purity)+'_'+str(entropy)+'_'+str(gen_iterations)+'.pth')
                    torch.save(netG.state_dict(), './model/32_linear_cell5n_netG_'+str(purity)+'_'+str(entropy)+'_'+str(gen_iterations)+'.pth')
                    torch.save(netD_D.state_dict(), './model/32_linear_cell5n_netD_D_'+str(purity)+'_'+str(entropy)+'_'+str(gen_iterations)+'.pth')
                    torch.save(netD_Q.state_dict(), './model/32_linear_cell5n_netD_Q_'+str(purity)+'_'+str(entropy)+'_'+str(gen_iterations)+'.pth')