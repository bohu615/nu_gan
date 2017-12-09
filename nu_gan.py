from utils.networks_gan import train
from utils.segmentation import nuclei_segmentation
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='cell_representation', help='cell_representation | image_classification')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--rand', type=int, default=32, help='random variable dimensions')
parser.add_argument('--dis_category', type=int, default=5, help='categorical variable dimensions')
opt = parser.parse_args()
print(opt)

if opt.task == 'cell_representation':
  array = np.load('./data/nuclei.npy')
  label = np.load('./data/label.npy')
  train(array, array, label, batchsize=opt.batch_size, rand=opt.rand, dis_category=opt.dis_category)

'''
if opt.task = 'image_classification':
  normal_root = './data/normal/'
  nuclei_segmentation(normal_root, normal)
  abnormal_root = './data/abnormal/'
  nuclei_segmentation(abnormal_root, abnormal)
  
  array = np.load('./data/nuclei.npy')
  label = np.load('./data/label.npy')
  train(array, array, label, batchsize=batch_size, rand=rand, dis_category=dis_category)
'''
  