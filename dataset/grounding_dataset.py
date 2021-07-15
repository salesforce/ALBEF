import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption

class grounding_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, mode='train'):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.mode = mode
        
        if self.mode == 'train':
            self.img_ids = {} 
            n = 0
            for ann in self.ann:
                img_id = ann['image'].split('/')[-1]
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1            
        

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])            
        image = Image.open(image_path).convert('RGB')  
        image = self.transform(image)
        
        caption = pre_caption(ann['text'], self.max_words) 
        
        if self.mode=='train':
            img_id = ann['image'].split('/')[-1]

            return image, caption, self.img_ids[img_id]
        else:
            return image, caption, ann['ref_id']