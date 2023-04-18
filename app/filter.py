from PIL import Image
import numpy as np

from converter import Converter
from correlator import Correlator

class Filter:
    def __init__(self):
        self.image = None
        self.output = None

    def apply_negative_filter(self, image_path, R=False,G=False,B=False):
        image = np.array(Image.open(image_path).convert('RGB'))

        if R:
            image[:,:,0] = 255 - image[:,:,0]

        if G:
            image[:,:,1] = 255 - image[:,:,1]

        if B:
            image[:,:,2] = 255 - image[:,:,2]

        transf_image = Image.fromarray(image.astype('uint8'))
        return transf_image
    
    def apply_negative_filter_in_y(self, image_path):
        converter = Converter()

        _, yiq_arr = converter.RGB_2_YIQ(image_path=image_path)

        yiq = yiq_arr.copy()

        yiq[:,:,0] = 255 - yiq[:,:,0]

        rgb_img, _ =  converter.YIQ_2_RGB(arr_img=yiq)

        rgb = rgb_img.copy()
        return rgb
    

    def apply_median_filter(self, image_path, filter_shape=(3,3), zero_padding=True):
        c = Correlator()

        self.image = np.array(Image.open(image_path).convert('RGB'))

        c.image = self.image

        vertical_padding = filter_shape[0]//2
        horizontal_padding = filter_shape[1]//2

        if not horizontal_padding and not vertical_padding:
            print("Could not execute padding due to filter shape. Try a Bi dimensional kernel.")
            zero_padding = False

        if zero_padding:
            preprocessed_img = c.padding(horizontal_padding, vertical_padding)
            output = np.zeros((self.image.shape[0], self.image.shape[1], 3))
        else:
            preprocessed_img = self.image
            output = np.zeros((self.image.shape[0] - 2 * vertical_padding, self.image.shape[1] - 2 * horizontal_padding, 3))

        for i in range(preprocessed_img.shape[0] - filter_shape[0]):
            for j in range(preprocessed_img.shape[1] - filter_shape[1]):
                for k in range(3):
                    output[i,j,k] = np.median(preprocessed_img[i: i + filter_shape[0], j: j + filter_shape[1], k])
        
        return self.image, preprocessed_img, output