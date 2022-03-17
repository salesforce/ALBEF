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

class mvsa_dataset(Dataset):
    def __init__(self, data_root, transform, split):
        self.im_root = os.path.join(
            data_root,
            'image'
        )
        self._entry = _load_annotations(
            os.path.join(
                data_root,
                split + '.jsonline'
            )
        )
        self.transform = transform
        
    def __len__(self):
        return len(self._entry)
    
    def __getitem__(self, index):    
        label = int(self._entry[index]['label'])
        im_id = str(self._entry[index]['id'])
        image_path = os.path.join(self.im_root, im_id + '.jpg')  
            
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
            
        text = pre_ac(self._entry[index]['text'])            
        return image, text, label
