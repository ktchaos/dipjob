from PIL import Image
import numpy as np

from math import sqrt

from converter import Converter
from correlator import Correlator


class Filter:
    def __init__(self):
        self.image = None
        self.output = None

    def apply_negative_filter(self, image_path, R=False, G=False, B=False):
        # Transforma a imagem
        image = np.array(Image.open(image_path).convert('RGB'))

        # Subtrai cada valor da cor de 255 em cada uma das bandas de acordo com a flag desejada
        if R:
            image[:, :, 0] = 255 - image[:, :, 0]

        if G:
            image[:, :, 1] = 255 - image[:, :, 1]

        if B:
            image[:, :, 2] = 255 - image[:, :, 2]

        transf_image = Image.fromarray(image.astype('uint8'))
        return transf_image

    def apply_negative_filter_in_y(self, image_path):
        converter = Converter()

        # Converte de RGB para YIQ
        _, yiq_arr = converter.RGB_2_YIQ(image_path=image_path)

        # Faz uma cópia do array que foi resultado do processo acima
        yiq = yiq_arr.copy()

        # Negativo!
        yiq[:,:,0] = 255 - yiq[:,:,0]

        # Converte de volta de YIQ para RGB
        rgb_img, _ = converter.YIQ_2_RGB(arr_img=yiq)

        # Faz uma copia da imagem e retorna
        rgb = rgb_img.copy()
        return rgb

    def apply_median_filter(self, image_path, filter_shape=(3,3), zero_padding=True):
        # Necessario um correlator para aplicar a correlação
        c = Correlator()

        # Transforma em um array de RGB
        self.image = np.array(Image.open(image_path).convert('RGB'))

        # Atribui a imagem em questao ao correlator
        c.image = self.image

        # Calcula a quantidade de linhas e colunas que precisa ser adicionado
        vertical_padding = filter_shape[0]//2
        horizontal_padding = filter_shape[1]//2

        # Se não tem nenhuma linha ou coluna para adicionar, significa que não é um array bidimensional
        if not horizontal_padding and not vertical_padding:
            print("Não foi possível fazer a extensão por zeros, tente com uma máscara bidimensional.")
            zero_padding = False

        # Se tiver extensão por zero, adiciona na variavel preprocessed_img
        if zero_padding:
            preprocessed_img = c.padding(horizontal_padding, vertical_padding)
            output = np.zeros((self.image.shape[0], self.image.shape[1], 3))
        else:
            preprocessed_img = self.image
            output = np.zeros(
                (self.image.shape[0] - 2 * vertical_padding, self.image.shape[1] - 2 * horizontal_padding, 3))

        # Deslizando a máscara na imagem:
        # Primeiro for é para percorrer em X
        for i in range(preprocessed_img.shape[0] - filter_shape[0]):
            # Segundo for é para percorrer em Y
            for j in range(preprocessed_img.shape[1] - filter_shape[1]):
                for k in range(3):
                    # o pixel em questao
                    output[i,j,k] = np.median(preprocessed_img[i: i + filter_shape[0], j: j + filter_shape[1], k])
        
        return self.image, preprocessed_img, output

    def apply_sobel_normalized_filter(self, root_path, image_path, zero_padding=True):
        c = Correlator()

        # x
        filter_mask_x = np.loadtxt(
            root_path + "/filters/sobel-x.txt", encoding=None, delimiter=",")
        mask_x = np.array(filter_mask_x)

        _, _, output = c.apply_correlation(
            image_path=image_path, filter_matrix=mask_x, zero_padding=zero_padding)
        output_x = output
        del output

        # y
        filter_mask_y = np.loadtxt(
            root_path + "/filters/sobel-y.txt", encoding=None, delimiter=",")
        mask_y = np.array(filter_mask_y)

        _, _, output = c.apply_correlation(
            image_path=image_path, filter_matrix=mask_y, zero_padding=zero_padding)
        output_y = output
        del output

        # new image
        h = output_x.shape[0]
        w = output_x.shape[1]
        output = np.zeros((h, w, 3))

        for i in range(h):
            for j in range(w):
                for k in range(3):
                    value = sqrt(
                        output_x[i, j, k] ** 2 + output_y[i, j, k] ** 2
                    )
                    value = int(min(value, 255))
                    output[i, j, k] = value

        return output

    def apply_box_one_dimensional_filter(self, root_path, image_path, zero_padding=True):
        c = Correlator()

        filter_box_1 = np.loadtxt(
            root_path + "/filters/box_11_1.txt", encoding=None, delimiter=",")
        matrix_1 = np.array(filter_box_1)
        mask_1 = matrix_1/matrix_1.size

        _, _, output = c.apply_correlation_one_dimensional(
            filter_matrix=mask_1, zero_padding=zero_padding, image_path=image_path)
        output_1 = output

        filter_box_2 = np.loadtxt(
            root_path + "/filters/box_1_11.txt", encoding=None, delimiter=",")
        matrix_2 = np.array(filter_box_2)
        mask_2 = matrix_2/matrix_2.size

        _, _, output = c.apply_correlation_one_dimensional(
            filter_matrix=mask_2, zero_padding=zero_padding, arr_img=output_1)
        output_2 = output

        return output_2
