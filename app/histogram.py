from PIL import Image
import numpy as np

class Histogram:
    def expansion(self, image, L=255):

        image_array = np.array(image)

        h = image_array.shape[0]
        w = image_array.shape[1]
        r_min = np.amin(image_array)
        r_max = np.amax(image_array)

        output = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                r = image_array[i, j]
                s = round(
                    ((r - r_min) / (r_max - r_min)) * (L - 1)
                )
                output[i, j] = s

        return output