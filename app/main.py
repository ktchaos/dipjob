import numpy as np
from PIL import Image

from converter import Converter
from filter import Filter
from correlator import Correlator

### QUESTAO 1) Conversão RGB-YIQ-RGB
def primeira_questao():
    # 1. Carregar imagem de teste
    orig_image = Image.open('/Users/catarinaserrano/Desktop/UFPB/PDI-TP-01/app/assets/dancer.jpg')
    orig_image.show()

    # 2. Aplicar a conversão RGB para YIQ
    converter = Converter()

    yiq_image , yiq_image_array = converter.RGB_2_YIQ(image_obj=orig_image)
    yiq_image.show()

    #3. Aplicar conversão YIQ para RGB
    rgb_image , _ = converter.YIQ_2_RGB(yiq_image_array)
    rgb_image.show()


### QUESTAO 2) Filtros negativos
def segunda_questao():
    image_path = '/Users/catarinaserrano/Desktop/UFPB/PDI-TP-01/app/assets/dancer.jpg'
    original_image = Image.open(image_path)
    original_image.show()
    
    filter = Filter()
    
    ## 1. Em R:
    negative_R_image = filter.apply_negative_filter(image_path=image_path, R=True)
    #negative_R_image.show()
    
    ## 2. Em G:
    negative_G_image = filter.apply_negative_filter(image_path=image_path, G=True)
    #negative_G_image.show()

    ## 3. Em B:
    negative_B_image = filter.apply_negative_filter(image_path=image_path, B=True)
    #negative_B_image.show()

    ## 4. Na banda Y
    negative_Y_image = filter.apply_negative_filter_in_y(image_path=image_path)
    #negative_Y_image.show()

## QUESTAO 3) Filtros com extensão por zeros
def terceira_questao():
    image_path = '/Users/catarinaserrano/Desktop/UFPB/PDI-TP-01/app/assets/lenaverso.jpg'
    original_image = Image.open(image_path)
    original_image.show()

    # Aplicando filtro da média (box)
    c = Correlator()
    filter_mask = np.loadtxt("/Users/catarinaserrano/Desktop/UFPB/PDI-TP-01/app/filters/media.txt", encoding=None, delimiter=",")
    matrix = np.array(filter_mask)
    mask = matrix/matrix.size

    _, _, output = c.apply_correlation(image_path=image_path, filter_matrix=mask, zero_padding=True)
    transf_img = Image.fromarray(output.astype('uint8'))
    transf_img.show()



## QUESTAO 4) Filtro da MEDIANA sobre R, G e B
def quarta_questao():
    image_path = '/Users/catarinaserrano/Desktop/UFPB/PDI-TP-01/app/assets/dancer.jpg'
    original_image = Image.open(image_path)
    original_image.show()

    filter = Filter()
    _, _, output = filter.apply_median_filter(image_path=image_path, filter_shape=(1, 1), zero_padding=False)
    tranf_image = Image.fromarray(output.astype('uint8'))
    tranf_image.show()