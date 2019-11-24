import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import math
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn.functional as F
from data_loader import LandmarksDataset, GrayscaleToRGB
from model import LandmarkModel, ArcMarginProduct, SimpleFC
import torchvision.utils as vutils
from batchHardTripletSelector import BatchHardTripletSelector, pdist
from visdom import Visdom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisdomLinePlotter(object):
  def __init__(self, env_name='main'):
    self.viz = Visdom()
    self.env = env_name
    self.plots = {}

  def update(self, env, plots):
    self.env = env
    self.plots = plots
    for var_name in self.plots:
      self.plots[var_name]
    
  def plot(self, var_name, split_name, title_name, x_label, x, y):
    if var_name not in self.plots:
      self.plots[var_name] = self.viz.line(X=np.array([x,x]),
                  Y=np.array([y,y]),
                  env=self.env, opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel=x_label,
                    ylabel=var_name
                  ))
    else:
      self.viz.line(X=np.array([x]), Y=np.array([y]), 
            env=self.env, win=self.plots[var_name],
            name=split_name, update = 'append')

def train(args):
  K = args.num_img_per_class
  batch_size = args.batch_size

  train_dataset = LandmarksDataset(csv_file=args.train_files,
                                   root_dir=args.images_dir,
                                   image_size=args.image_size,
                                   K = K,
                                   mode="train")
  val_dataset = LandmarksDataset(csv_file=args.val_files,
                                 root_dir=args.images_dir,
                                 image_size=args.image_size,
                                 K = args.eval_K,
                                 mode = "val")

  train_sampler = torch.utils.data.WeightedRandomSampler(
      weights=train_dataset.weights_per_line,
      num_samples=train_dataset.total_images//K,
      replacement=True)
  val_sampler = torch.utils.data.WeightedRandomSampler(
      weights=val_dataset.weights_per_line,
      num_samples=val_dataset.total_images//args.eval_K,
      replacement=True)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=16)

  landmarkModel = LandmarkModel().to(device)
  print(landmarkModel)
  
  lr = args.lr
  optim = torch.optim.Adam(params=landmarkModel.parameters(),lr=lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optim, args.decay_every, gamma=args.learning_anneal)

  start_epoch = 0
  end_epoch = args.num_epochs
  selector = BatchHardTripletSelector()
  best_val_loss = float('inf')

  if args.num_classes:
    num_classes = args.num_classes
  else:
    num_classes = len(train_dataset)

  print(num_classes)
  metric_fc = ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin=False).to(device)
  # print(num_classes)
  # simplefc = SimpleFC(512, num_classes).to(device)
  criterion1 = torch.nn.CrossEntropyLoss().to(device)
  criterion2 = nn.TripletMarginLoss(margin=args.triplet_margin, p=2).to(device)
  alpha = args.alpha
  beta = args.beta

  if args.load_model:
    checkpoint = torch.load(args.load_path)
    landmarkModel.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    metric_fc.weight = checkpoint["arcFace"]
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    # plotter.env, plotter.plots = checkpoint["visdom_plots"]
    plotter.update(checkpoint["visdom_plots"][0], checkpoint["visdom_plots"][1])

    # with torch.no_grad():
      # val_loss, val_accuracy = eval(val_loader, landmarkModel, args.eval_K, selector,
      #                                   criterion1, criterion2, alpha, metric_fc)
  
  total_steps = 0
  for epoch in range(start_epoch,end_epoch):

    landmarkModel.train()
    start_time = time.time()
    avg_loss = 0.0
    total_correct = 0
    total_samples = 0
    step_loss1 = 0
    step_loss2 = 0
    step_accuracy = 0

    for step, sample in enumerate(train_loader):
      total_steps += 1

      inp = sample['images'].to(device)
      inp = inp.view(-1,inp.shape[-3],inp.shape[-2],inp.shape[-1])
      labels = sample['landmark_id'].reshape(-1,1).repeat(1,K).reshape(-1,1).to(device)

      plt.figure(figsize=(4,4))
      plt.axis("off")
      plt.title("Training Images")
      plt.imshow(np.transpose(vutils.make_grid(inp[:16], padding=2, normalize=True).cpu(),(1,2,0)))
      plt.savefig("data.png")

      embds = landmarkModel(inp)
      output = metric_fc(embds, labels)
      # output = simplefc(embds)
      # dist_mtx = pdist(embds, embds)
      # print(dist_mtx)
      anchor, pos, neg = selector(embds, labels)
        
      loss1 = criterion1(output, labels.view(-1))
      loss2 = criterion2(anchor, pos, neg)
      loss = alpha*loss1 + beta*loss2

      predicted_labels = torch.argmax(output, dim=1)
      # print(predicted_labels)
      # print(labels.view(-1))
      correct_predictions = torch.sum((predicted_labels == labels.view(-1))).item()
      batch_accuracy = (correct_predictions / labels.shape[0]) * 100
      total_correct += correct_predictions
      total_samples += labels.shape[0]
      step_loss1 += loss1.item()
      step_loss2 += loss2.item()
      step_accuracy += batch_accuracy

      optim.zero_grad()
      loss.backward()
      optim.step()
      
      avg_loss += loss.item()
      print("Epoch {}, step {}/{}, lr {:.8f}, classification loss {:.3f}, triplet loss {:.3f}, loss {:.3f}, accuracy {:.5f}"
            .format(epoch, step, len(train_loader), scheduler.get_lr()[0], loss1.item(), loss2.item(), loss.item(), batch_accuracy))
      if step % 100 == 0 and step > 0: 
        # plotter.plot('arcFace loss', 'arcFace loss', 'ArcFace Loss vs steps', 'Step', total_steps, step_loss1/100)
        # plotter.plot('triplet loss', 'triplet loss', 'Triplet Loss vs steps', 'Step', total_steps, step_loss2/100)
        # plotter.plot('step Accuracy', 'train', 'Accuracy vs steps', 'Step', total_steps, step_accuracy/100)
        step_loss1 = 0 
        step_loss2 = 0
        step_accuracy = 0

    avg_loss = avg_loss/len(train_loader)
    total_accuracy = total_correct/total_samples * 100
    epoch_time = time.time() - start_time
    print('Training Summary Epoch: [{0}]\t'
          'Time taken (s): {epoch_time:.0f}\t'
          'Average Loss {loss:.8f}\t'
          'Accuracy {accuracy}'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss, accuracy=total_accuracy))
    plotter.plot('loss', 'train', 'Loss vs epoch', 'Epoch', epoch, avg_loss)
    plotter.plot('Accuracy', 'train', 'Accuracy vs epoch', 'Epoch', epoch, total_accuracy)
    avg_loss = 0.0
    total_correct = 0
    total_samples = 0
    scheduler.step()

    if epoch % 10 == 0:
      save_path = os.path.join(args.save_dir, "model_{}.pth".format(epoch))
      print('Saving model to {}'.format(save_path))
      torch.save({
          'epoch' : epoch + 1,
          'model_state_dict': landmarkModel.state_dict(),
          'optimizer_state_dict': optim.state_dict(),
          'scheduler_state_dict' : scheduler.state_dict(),
          'best_val_loss' : best_val_loss,
          'arcFace' : metric_fc.weight,
          'visdom_plots' : (plotter.env, plotter.plots),
      }, save_path)
    
    with torch.no_grad():
      val_loss, val_accuracy = eval(val_loader, landmarkModel, args.eval_K, selector,
                                    criterion1, criterion2, alpha, metric_fc)
      plotter.plot('loss', 'val', 'Loss vs epoch', 'Epoch', epoch, val_loss)
      plotter.plot('Accuracy', 'val', 'Accuracy vs epoch', 'Epoch', epoch, val_accuracy)

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_path = os.path.join(args.save_dir, "model_final.pth")
      print("Saving best model to {}".format(save_path))
      torch.save({
        'epoch' : epoch,
        'model_state_dict': landmarkModel.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'best_val_loss' : best_val_loss,
        'arcFace' : metric_fc.weight,
        'visdom_plots' : (plotter.env, plotter.plots),
      }, save_path)

# def eval(val_loader, model, K, selector, criterion):
def eval(val_loader, model, K, selector, criterion1, criterion2, alpha, metric_fc):
  model.eval()
  start_time = time.time()
  avg_loss = 0.0
  total_correct = 0
  total_samples = 0
  for step, sample in tqdm(enumerate(val_loader), total=math.ceil(len(val_loader))):
    inp = sample['images'].to(device)
    inp = inp.view(-1,inp.shape[-3],inp.shape[-2],inp.shape[-1])
    labels = sample['landmark_id'].reshape(-1,1).repeat(1,K).reshape(-1,1).to(device)
    
    embds = model(inp)
    anchor, pos, neg = selector(embds, labels)
    output = metric_fc(embds, labels)
    loss1 = criterion1(output, labels.view(-1))
    loss2 = criterion2(anchor, pos, neg)
    loss = alpha*loss1 + loss2
    avg_loss += loss.item()
    predicted_labels = torch.argmax(output, dim=1)
    correct_predictions = torch.sum(predicted_labels == labels.view(-1)).item()
    batch_accuracy = (correct_predictions / labels.shape[0]) * 100
    total_correct += correct_predictions
    total_samples += labels.shape[0]
    
  avg_loss = avg_loss/len(val_loader)
  total_accuracy = total_correct / total_samples * 100
  epoch_time = time.time() - start_time
  print('-'*80)
  print("Validation loss {}. accuracy {:.5f}".format(avg_loss, total_accuracy))
  return avg_loss, total_accuracy

def extract_feature_discriptors(args):

  landmarkModel = LandmarkModel().to(device)
  landmarkModel.eval()
  checkpoint = torch.load(args.load_path)
  landmarkModel.load_state_dict(checkpoint['model_state_dict'])
  with open(args.train_files,'r') as f:
    lines = f.readlines()

  image_size = args.image_size
  transform_list = [
      transforms.Resize(image_size),
      transforms.CenterCrop(image_size),
      transforms.ToTensor(),
      GrayscaleToRGB(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])]

  transform = transforms.Compose(transform_list)

  image_embds = []
  for line in tqdm(lines, total=len(lines)):
    line = line.strip().split(',')
    images = line[1].split()
    for img in images:
      img_path = os.path.join(args.images_dir, img)
      image_tensor = transform(Image.open(img_path)).unsqueeze(0).to(device)
      embd = landmarkModel(image_tensor).squeeze()
      image_embds.append((img,embd.detach().cpu().numpy()))

  print(len(image_embds))
  np.save("train_embds.npy",image_embds)

def get_topk_matches(args, k=5):

  print("Loading trained model from {}".format(args.load_path))
  landmarkModel = LandmarkModel().to(device)
  print("Loaded trained model")
  landmarkModel.eval()
  checkpoint = torch.load(args.load_path)
  landmarkModel.load_state_dict(checkpoint['model_state_dict'])

  image_size = args.image_size
  transform_list = [
      transforms.Resize(image_size),
      transforms.CenterCrop(image_size),
      transforms.ToTensor(),
      GrayscaleToRGB(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])]

  transform = transforms.Compose(transform_list)

  train_image_embds = np.load(args.embd_path, allow_pickle=True)
  test_img_path = args.img_path
  image_tensor = transform(Image.open(test_img_path)).unsqueeze(0).to(device)
  test_img_embd = landmarkModel(image_tensor).squeeze().detach().cpu().numpy()
  
  dist_vec = []
  for img in train_image_embds:
    train_img_name = img[0]
    train_img_embd = img[1]
    dist = np.linalg.norm(test_img_embd - train_img_embd)
    dist_vec.append((train_img_name, dist))

  best_matches = sorted(dist_vec, key=lambda x:x[1])[:k]

  fig = plt.figure()
  ax1 = fig.add_subplot(1,k+1,1)
  ax1.axis("off")
  ax1.imshow(plt.imread(test_img_path))
  for i in range(k):
    ax2 = fig.add_subplot(1,k+1,i+2)
    ax2.axis("off")
    ax2.annotate(os.path.dirname(best_matches[i][0][4:]), xy=(0,-0.3), xycoords="axes fraction")
    ax2.imshow(plt.imread(os.path.join(args.images_dir, best_matches[i][0])))

  plt.savefig("matches.png", bbox_inches='tight')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs to train the model')
  parser.add_argument('--batch_size', type=int, default=64, help='')
  parser.add_argument('--image_size', type=int, default=224, help='')
  parser.add_argument('--num_classes', type=int, default=None, help='')
  parser.add_argument('--decay_every', type=int, default=1, help='')
  parser.add_argument('--eval_K', type=int, default=4, help='')
  parser.add_argument('--triplet_margin', type=float, default=1., help='margin for triplet loss')
  parser.add_argument('--alpha', type=float, default=0.1, help='weight for combining loss')
  parser.add_argument('--beta', type=float, default=1., help='weight for combining loss')
  parser.add_argument('--learning_anneal', type=float, default=0.95, help='')
  parser.add_argument('--num_img_per_class', type=int, default=4,
                       help='Number of samples belonging to each class per batch')
  parser.add_argument('--lr', type=float, default=0.0001, help='')
  parser.add_argument('--save_dir', type=str, default="saved_models", help='directory to save models using training')
  parser.add_argument('--load_model', action='store_true', help='continue training from a pretrained model')
  parser.add_argument('--load_path', type=str, default=None, help='directory to load pretrained model')
  parser.add_argument('--train_files', type=str, default="/home/kartik/exp/data/google_landmark_recognition/train.csv",
                       help='csv file path for train files')
  parser.add_argument('--val_files', type=str, default="/home/kartik/exp/data/google_landmark_recognition/val.csv",
                       help='csv file path for Validation files')
  parser.add_argument('--images_dir', type=str, default="/home/kartik/exp/data/google_landmark_recognition/images",
                       help='path to images directory')
  parser.add_argument('--extract_embeddings', action='store_true', help='extract embeddings from the train data')
  parser.add_argument('--get_topk_matches', action='store_true', help='get the closest_topk matches for the images')
  parser.add_argument('--embd_path', type=str, default=None, help='path to embeddings of the train images')
  parser.add_argument('--img_path', type=str, default=None, help='path to image for which we need closest matches')

  args = parser.parse_args()
  if args.extract_embeddings:
    extract_feature_discriptors(args)
  elif args.get_topk_matches:
    get_topk_matches(args)
  else:
    global plotter
    plotter = VisdomLinePlotter()
    train(args)
    