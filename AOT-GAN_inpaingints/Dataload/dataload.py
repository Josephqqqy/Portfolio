#### 用于加载数据，数据预处理，数据增强
import os
from glob import glob
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset



class InpaintingData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w , self.h = args.image_size,args.image_size
        self.mask_type = args.mask_type
        self.dir_image = args.dir_image
        self.dir_mask = args.dir_mask
        self.data_train = args.data_train
        self.mask_type = args.mask_type
        self.mask_names = ['l_brow','neck','u_lip','l_lip','eye_g','mouth','hat','ear_r','hair','r_eye','neck_l','r_brow','l_eye','l_ear','r_ear','cloth','nose']
        # image and mask
        self.image_path = []
        for ext in ["*.jpg", "*.png"]:
            self.image_path.extend(glob(os.path.join(self.dir_image, self.data_train, ext)))
        random.shuffle(self.image_path)
        self.mask_path = glob(os.path.join(self.dir_mask, self.mask_type, "*.png"))
        # augmentation
        self.img_trans = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                # transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor(),
            ]
        )
        self.mask_trans = transforms.Compose(
            [
                transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((0, 45), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )
    def get_mask(self,image_number):
        mask_exist = []
        for i in self.mask_names:
            mask_filenames = f"{str(image_number).zfill(5)}_{i}.png"
            mask_dir = os.path.join(self.dir_mask,self.mask_type, mask_filenames)  
            if os.path.exists(mask_dir):
                mask_exist.append(mask_dir)
        mask_filename = random.choice(mask_exist)
        mask = Image.open(mask_filename).convert("L")
        return mask
    
    def __len__(self):
        return len(self.image_path)
    def __iter__(self):
        yield 
    def __getitem__(self, index):
        # load image
        image = Image.open(self.image_path[index]).convert("RGB")  #打开image
        filename = os.path.basename(self.image_path[index])     #读取image的文件名
        # extract image number from filename
        image_number = int(filename.split('.')[0])        #提取文件名前缀

        if self.mask_type == "pconv":
            mask = self.get_mask(image_number)
        else:
            mask = np.zeros((self.h, self.w)).astype(np.uint8)
            mask[self.h // 4 : self.h // 4 * 3, self.w // 4 : self.w // 4 * 3] = 1
            mask = Image.fromarray(mask).convert("L")
        
        
        # augment
        image = self.img_trans(image) * 2.0 - 1.0
        mask = self.mask_trans(mask)

        return image , mask, filename