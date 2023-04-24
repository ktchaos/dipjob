import numpy as np
import time
from sys import argv
from PIL import Image

from converter import Converter
from filter import Filter
from correlator import Correlator
from histogram import Histogram


catarina_root_path = '/Users/catarinaserrano/Desktop/UFPB/PDI-TP-01/app'
arthur_root_path = '/Users/arthurruan/www/ufpb/dipjob/app'
anderson_root_path = '/Users/anderson.leite/Documents/Anderson/git/PDI/TP-01/dipjob/app'

root_profile = argv[1]
root_path = ''

match root_profile:
    case 'anderson':
        root_path = anderson_root_path
    case 'arthur':
        root_path = arthur_root_path
    case 'catarina':
        root_path = catarina_root_path
    case _:
        raise ValueError('root profile inválido!')

# QUESTAO 1) Conversão RGB-YIQ-RGB
def primeira_questao():
    # 1. Carregar imagem de teste
    orig_image = Image.open(root_path + '/assets/dancer.jpg')
    orig_image.show()

    # 2. Aplicar a conversão RGB para YIQ
    converter = Converter()

    yiq_image, yiq_image_array = converter.RGB_2_YIQ(image_obj=orig_image)
    yiq_image.show()

    # 3. Aplicar conversão YIQ para RGB (voltando)
    rgb_image, _ = converter.YIQ_2_RGB(yiq_image_array)
    rgb_image.show()


# QUESTAO 2) Filtros negativos
def segunda_questao():
    image_path = root_path + '/assets/dancer.jpg'
    original_image = Image.open(image_path)
    original_image.show()

    filter = Filter()

    # 1. Em R:
    negative_R_image = filter.apply_negative_filter(
        image_path=image_path, R=True)
    # negative_R_image.show()

    # 2. Em G:
    negative_G_image = filter.apply_negative_filter(
        image_path=image_path, G=True)
    # negative_G_image.show()

    # 3. Em B:
    negative_B_image = filter.apply_negative_filter(
        image_path=image_path, B=True)
    # negative_B_image.show()

    # 4. Na banda Y
    negative_Y_image = filter.apply_negative_filter_in_y(image_path=image_path)
    # negative_Y_image.show()

    # 5. Em RGB simultaneamente
    negative_rgb_image = filter.apply_negative_filter(image_path=image_path, R=True, G=True, B=True)
    # negative_rgb_image.show()


# QUESTAO 3) Filtros com extensão por zeros
def terceira_questao():
    image_path = root_path + '/assets/dancer.jpg'
    original_image = Image.open(image_path)
    original_image.show()

    # Aplicando Box11x11
    initial = initial = time.time()
    c_1 = Correlator()
    filter_mask = np.loadtxt(
        root_path + "/filters/box_11_11.txt", encoding=None, delimiter=",")
    matrix = np.array(filter_mask)
    mask = matrix/matrix.size
    _, _, output = c_1.apply_correlation(
        image_path=image_path, filter_matrix=mask, zero_padding=True)
    transf_img = Image.fromarray(output.astype('uint8'))
    transf_img.show()
    print("time taken Box11x11) = ", time.time()-initial)

    # Aplicando Box11x1(Box1X11(image))
    initial = initial = time.time()
    filter = Filter()
    output = filter.apply_box_one_dimensional_filter(
        root_path=root_path, image_path=image_path)
    transf_img = Image.fromarray(output.astype('uint8'))
    transf_img.show()
    print("time taken Box11x1(Box1X11(image)) = ", time.time()-initial)

    # Aplicando filtro da média (box)
    c_1 = Correlator()
    filter_mask = np.loadtxt(
        root_path + "/filters/media.txt", encoding=None, delimiter=",")
    matrix = np.array(filter_mask)
    mask = matrix/matrix.size

    _, _, output = c_1.apply_correlation(
        image_path=image_path, filter_matrix=mask, zero_padding=True)
    transf_img = Image.fromarray(output.astype('uint8'))
    transf_img.show()

    # Aplicando filtro Emboss
    c_2 = Correlator()
    filter_mask = np.loadtxt(
        root_path + "/filters/emboss.txt", encoding=None, delimiter=",")
    matrix = np.array(filter_mask)

    _, _, output = c_2.apply_correlation(
        image_path=image_path, filter_matrix=matrix, zero_padding=True)
    abs_img = np.abs(output)
    offset = 128
    filter_img = abs_img + offset
    transf_img = Image.fromarray(filter_img.astype('uint8'))
    transf_img.show()

    # Aplicando Sobel normalizado
    filter = Filter()
    output = filter.apply_sobel_normalized_filter(
        root_path=root_path, image_path=image_path)
    transf_img = Image.fromarray(output.astype('uint8'))
    transf_img.show()

    # Aplicando expansão de histograma no resultado do filtro de Sobel normalizado
    h = Histogram()
    gray_img = transf_img.convert('L')
    gray_img.show()
    output = h.expansion(image=gray_img, L=255)
    transf_img = Image.fromarray(output.astype('uint8'))
    transf_img.show()


# QUESTAO 4) Filtro da MEDIANA sobre R, G e B
def quarta_questao():
    image_path = root_path + '/assets/dancer.jpg'
    original_image = Image.open(image_path)
    original_image.show()

    filter = Filter()
    _, _, output = filter.apply_median_filter(
        image_path=image_path, filter_shape=(1, 1), zero_padding=False)
    tranf_image = Image.fromarray(output.astype('uint8'))
    tranf_image.show()


# execução principal do programa
question = argv[2]


def main():
    match question:
        case '1':
            primeira_questao()
        case '2':
            segunda_questao()
        case '3':
            terceira_questao()
        case '4':
            quarta_questao()
        case _:
            raise ValueError('Questão inválida!')


main()
