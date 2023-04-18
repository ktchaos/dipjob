import numpy as np
from PIL import Image

from converter import Converter

# 1. Carregar imagem de teste
orig_image = Image.open('/Users/catarinaserrano/Desktop/UFPB/PDI-TP-01/app/assets/dancer.jpg')
orig_image.show()

# 2. Aplicar a conversão RGB para YIQ
converter = Converter()

yiq_image , yiq_image_array = converter.RGB_2_YIQ(image_obj=orig_image)
yiq_image.show()

#3. Aplicar conversão YIQ para RGB
rgb_image , rgb_image_array = converter.YIQ_2_RGB(yiq_image_array)
rgb_image.show()
