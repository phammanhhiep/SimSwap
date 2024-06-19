'''
Modified version of data_loader_Swapping.py to handle different dataset
'''

import os
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class data_prefetcher():
  def __init__(self, loader):
    self.loader = loader
    self.dataiter = iter(loader)
    self.stream = torch.cuda.Stream()
    self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1)
    self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)
    self.num_images = len(loader)
    self.preload()

  def preload(self):
    """Load beforehand two images to use in the next iteration.  
    """
    try:
      self.src_image1, self.src_image2 = next(self.dataiter)
    except StopIteration:
      self.dataiter = iter(self.loader)
      self.src_image1, self.src_image2 = next(self.dataiter)
      
    with torch.cuda.stream(self.stream):
      self.src_image1  = self.src_image1.cuda(non_blocking=True)
      self.src_image1  = self.src_image1.sub_(self.mean).div_(self.std)
      self.src_image2  = self.src_image2.cuda(non_blocking=True)
      self.src_image2  = self.src_image2.sub_(self.mean).div_(self.std)

  def next(self):
    torch.cuda.current_stream().wait_stream(self.stream)
    src_image1  = self.src_image1
    src_image2  = self.src_image2
    self.preload()
    return src_image1, src_image2
  
  def __len__(self):
    """Return the number of images."""
    return self.num_images


class SwappingDataset(data.Dataset):
  """Dataset class for the Artworks dataset and content dataset."""

  def __init__(self,
          image_dir,
          img_transform,
          random_seed=1234):
    """Initialize and preprocess the Swapping dataset."""
    self.image_dir      = image_dir
    self.img_transform  = img_transform   
    self.dataset        = []
    self.random_seed    = random_seed
    self.preprocess()

  def preprocess(self):
    """
      Get list of all images from the root folder; consider all images
      in any nested folders. 
    """
    print("processing Swapping dataset images...")
    self.dataset = [
      os.path.join(path, filename)
      for path, dirs, files in os.walk(self.image_dir)
      for filename in files 
      if(filename.endswith(".png") or 
      filename.endswith(".jpg") or 
      filename.endswith(".jpeg"))
    ]
    random.seed(self.random_seed)
    random.shuffle(self.dataset)
    print('Finished preprocessing the Swapping dataset, total dirs number: %d...'%len(self.dataset))
       
  def __getitem__(self, index):
    """
    Mimic the original SwappingDataset in data_loader_Swapping.py, by return
    the same image. That suite for dataset, in which images of the same identity
    are not grouped in the same folder.
    """
    image_path = self.dataset[index]
    image1 = Image.open(image_path)
    image2 = image1.copy()

    image1 = self.img_transform(image1)
    image2 = self.img_transform(image2)

    return image1, image2
  
  def __len__(self):
    return len(self.dataset)


def GetLoader(  
  dataset_roots,
  batch_size=16,
  dataloader_workers=8,
  random_seed=1234,
  image_size=224,
  ):
  """Build and return a data loader."""
    
  num_workers         = dataloader_workers
  data_root           = dataset_roots
  random_seed         = random_seed
  
  c_transforms = T.Compose([
    T.ToTensor(),
    T.Resize(image_size)
  ])

  content_dataset = SwappingDataset(
    data_root, 
    c_transforms,
    random_seed)
  content_data_loader = data.DataLoader(
    dataset=content_dataset,
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True)
  prefetcher = data_prefetcher(content_data_loader)
  
  return prefetcher


def denorm(x):
  out = (x + 1) / 2
  return out.clamp_(0, 1)