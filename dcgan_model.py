from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf 
import numpy as np 
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size,stride) :
	return int(math.ceil(float(size) / float(stride)))

class DCGAN(object) :
	def __init__(self, sess, input height=108, input_width=108, crop=True,
		batch_size = 64, sample_num=64, output_height=64, output_width=64,
		y_dim=None, z_dim=100, gf_dim=64, df=64,
		gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
		input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, data_dir='./data') :

		self.sess = sess
		self.crop = crop

		self.batch_size = batch_size
		self.sample_num = sample_num

		self.input_height = input_height
		self.input_width = input_width
		self.output_height = output_height
		self.output_width = output_width

		self.y_dim = y_dim
		self.z_dim = z_dim

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.gfc_dim = gfc_dim
		self.dfc_dim = dfc_dim
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')

		


