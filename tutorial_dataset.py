import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        data_path_promt = './CycleNet/cfill50k/prompt.json'
        data_path_trainA = './CycleNet/apple2orange/trainA.json'
        with open(data_path_trainA , 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_filename = item['image']
        source = item['source']
        target = item['target']
        cfill = 'CycleNet/cfill50k/' 
        trainA = 'CycleNet/apple2orange/' 

        image = cv2.imread(trainA + image_filename)

        # Do not forget that OpenCV read images in BGR order.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = (image.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=image, source=source, txt=target)

