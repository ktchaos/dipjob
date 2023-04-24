import numpy as np
from PIL import Image


class Correlator:

    def __init__(self):
        self.image = None
        self.filter = None

    def padding(self, horizontal_padding, vertical_padding):
        padded_image = np.zeros((self.image.shape[0] + 2 * vertical_padding, self.image.shape[1] + 2 * horizontal_padding, 3))
        
        # Verifica os casos que não precisa adicionar verticalmente ou horizontalmente e centraliza a imagem
        if vertical_padding == 0:
            padded_image[:, horizontal_padding: -
                         horizontal_padding, :] = self.image
        elif horizontal_padding == 0:
            padded_image[vertical_padding : -vertical_padding, :, :] = self.image
        else:
            padded_image[vertical_padding : -vertical_padding, horizontal_padding : -horizontal_padding, :] = self.image 
    
        # Retorna a imagem com a extensão por 0
        return padded_image

    def apply_correlation(self, image_path, filter_matrix, zero_padding=True):
        self.image = np.array(Image.open(image_path).convert('RGB'))
        self.filter = filter_matrix
        
        # numero de colunas que precisa ter para adicionar
        vertical_padding = self.filter.shape[0]//2
        # numero de linhas que precisa ter
        horizontal_padding = self.filter.shape[1]//2

        if not horizontal_padding and not vertical_padding:
            print(
                "Could not execute padding due to filter shape. Try a Bi dimensional kernel.")
            zero_padding = False

        if zero_padding:
            preprocessed_img = self.padding(
                horizontal_padding, vertical_padding)
            output = np.zeros((self.image.shape[0], self.image.shape[1], 3))
        else:
            preprocessed_img = self.image
            output = np.zeros(
                (self.image.shape[0] - 2 * vertical_padding, self.image.shape[1] - 2 * horizontal_padding, 3))

        for i in range(preprocessed_img.shape[0] - self.filter.shape[0]):
            for j in range(preprocessed_img.shape[1] - self.filter.shape[1]):
                for k in range(3):
                    output[i, j, k] = np.sum(np.multiply(
                        self.filter, preprocessed_img[i: i + self.filter.shape[0], j: j + self.filter.shape[1], k]))

        output[output < 0] = 0
        output[output > 255] = 255

        return self.image, preprocessed_img, output

    def apply_correlation_one_dimensional(self, filter_matrix, zero_padding=True, image_path=None, arr_img=None):
        if arr_img is None:
            self.image = np.array(Image.open(image_path).convert('RGB'))
        else:
            self.image = arr_img.copy()

        self.filter = filter_matrix

        vertical_padding = len(self.filter)//2
        horizontal_padding = 0

        if zero_padding:
            preprocessed_img = self.padding(
                horizontal_padding, vertical_padding)
            output = np.zeros((self.image.shape[0], self.image.shape[1], 3))
        else:
            preprocessed_img = self.image
            output = np.zeros(
                (self.image.shape[0] - 2 * vertical_padding, self.image.shape[1], 3))
        for i in range(preprocessed_img.shape[0] - len(self.filter)):
            for j in range(preprocessed_img.shape[1]):
                for k in range(3):
                    output[i, j, k] = np.sum(np.multiply(
                        self.filter, preprocessed_img[i: i + len(self.filter), j, k]))
        output[output < 0] = 0
        output[output > 255] = 255
        return self.image, preprocessed_img, output
