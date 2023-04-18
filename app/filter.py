from PIL import Image
import numpy as np

from converter import Converter

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