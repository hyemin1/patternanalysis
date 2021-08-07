from PIL import Image
import numpy as np
import os
from tqdm import tqdm
class mnist:
    def load_mnist(test_set=False, selection=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
        subset = 'testing' if test_set is True else 'training'
        base_path = os.path.join('mnist_png', subset)
        data = []
        lbl  = []
        for subfolder in tqdm(os.listdir(base_path), desc='Loading MNIST'):
            folder_path = os.path.join(base_path, subfolder)
            label = int(subfolder)
            if not label in selection:
                continue
            for im_name in os.listdir(folder_path):
                im_path = os.path.join(folder_path, im_name)
                im = Image.open(im_path).convert('L')
                im_array = np.array(im) # work on this array to modify the image
                data.append(im_array.reshape(-1))
                lbl.append(label)
        return np.array(data), np.array(lbl)
