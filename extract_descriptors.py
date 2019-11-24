import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import math
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn.functional as F
from data_loader import LandmarksDataset
from model import LandmarkModel, ArcMarginProduct, SimpleFC
import torchvision.utils as vutils
from batchHardTripletSelector import BatchHardTripletSelector, pdist
from visdom import Visdom

def extract_descriptors():
		birds_dataset = datasets.ImageFolder(
								root = "/home/kartik/exp/data/CUB_200_2011/")