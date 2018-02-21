import torch.utils.data as data

from .cityscapes_loader import *

class SemiCityscapes(cityscapesLoader):
  def __init__(self, root, split='train', is_transform=False,
               img_size=(512, 1024), augmentations=None, gamma_augmentation=0,
               city_names='*', real_synthetic='real'):
    super(SemiCityscapes, self).__init__(root, split, is_transform, img_size, augmentations,
                                         gamma_augmentation, city_names)
