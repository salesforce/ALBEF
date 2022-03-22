import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_ac
import jsonlines
import torch
import numpy as np

def _load_annotations(annotations_jsonpath):
    entries = []

    with open(annotations_jsonpath, "r", encoding="utf8") as f:
        for annotation in jsonlines.Reader(f):
            entries.append(
                {
                    "label": annotation["Rating"],
                    "id": annotation["_id"],
                    "text": annotation["Text"],
                    "photos": annotation["Photos"],
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
        
        self._max_num_img = 1

    def __len__(self):
        return len(self._entry)
    
    def __getitem__(self, index):    
        entry = self._entry[index]
        label = int(entry['label']) - 1

        im_s = torch.zeros(self._max_num_img, 3, 384, 384)
        cnt = 0
        try:
            photos = entry['photos']
            for im in photos:
                im_id = im['_id']
                image_path = os.path.join(
                    self.im_root,
                    im_id + '.jpg'
                )
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')   
                    image = self.transform(image)
                    im_s[cnt] = image
                    cnt += 1
                    if cnt == self._max_num_img:
                        break
        except KeyError:
            pass

        text = pre_ac(self._entry[index]['text'])
        return im_s, text, label
