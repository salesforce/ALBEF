import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_ac
import jsonlines

def _load_annotations(annotations_jsonpath):
    entries = []

    with open(annotations_jsonpath, "r", encoding="utf8") as f:
        for annotation in jsonlines.Reader(f):
            entries.append(
                {
                    "label": annotation["Rating"],
                    "id": annotation["_id"],
                    "text": annotation["Text"]
                }
            )
    return entries

class yelp_dataset(Dataset):
    def __init__(self, data_root, transform, split):
        self.im_root = os.path.join(
            data_root,
            'image'
        )
        self._entry = _load_annotations(
            os.path.join(
                data_root,
                split + '.json'
            )
        )
        self.transform = transform
        
    def __len__(self):
        return len(self._entry)
    
    def __getitem__(self, index):    
        entry = self._entry[index]
        label = int(entry['label']) - 1
        try:
            photos = entry['Photos']
            for im in photos:
                im_id = im['_id']
                image_path = os.path.join(
                    self.im_root,
                    im_id + '.jpg'
                )
                if os.path.exists(image_path):
                    break
                
            image = Image.open(image_path).convert('RGB')   
        except KeyError:
            image = Image.new('RGB', (256, 256), (255, 255, 255))
        image = self.transform(image)          
            
        text = pre_ac(self._entry[index]['text'])
        return image, text, label
