import numpy as np
from PIL import Image

class Converter:
    
    def __init__(self):
        self.image = None
        self.image_transformed = None
        
    def RGB_2_YIQ(self, image_path = None, image_obj = None):
        """
        Y: 0.299 * R + 0.587 * G + 0.114 * B
        I: 0.596 * R -  0.274 * G - 0.322 * B
        Q: 0.211 * R - 0.523 * G + 0.312 * B
        """
        if image_path != None:
            # Abre a imagem e transforma em um array do numpy
            image = Image.open(image_path).convert("RGB")
            self.image = Image.open(image_path).convert("RGB")
        elif image_obj != None:
            image = image_obj.copy()

        arr_img = np.asarray(image)
        
        # Faz uma cópia do array anterior mas transformado em float
        img_copy = arr_img.copy().astype(float)

        # Matriz YIQ
        matrix_yiq = np.array([[0.299, 0.587, 0.114],
                      [0.596, -0.274, -0.322],
                      [0.211, -0.523, 0.312]])
        
        # Operador com ponto (.dot)
        
        #img_copy[:,:,0] = arr_img[:,:,0] * 0.299 + arr_img[:,:,1] * 0.587 + arr_img[:,:,2] * 0.114
        #img_copy[:,:,1] = arr_img[:,:,0] * 0.596 + arr_img[:,:,1] * -0.274 + arr_img[:,:,2] * -0.322
        #img_copy[:,:,2] = arr_img[:,:,0] * 0.211 + arr_img[:,:,1] * -0.523 + arr_img[:,:,2] * 0.312
        
        img_copy = np.dot(arr_img, matrix_yiq.T.copy())
        
        # Transforma o array em imagem para retorno da função
        img_transformed = Image.fromarray(img_copy.astype('uint8'))
        self.image_transformed = Image.fromarray(img_copy.astype('uint8'))
        
        # Retorna a imagem e o array da imagem
        return img_transformed, img_copy
    
    def YIQ_2_RGB(self, arr_img):
        """
        R: 1.0 * Y + 0.956 * I + 0.621 * Q
        G: 1.0 * Y – 0.272 * I – 0.647 * Q
        B: 1.0 * Y – 1.106 * I + 1.703 * Q
        """ 
     
        # Matriz RGB
        matrix_rgb = np.array([
                               [1.0, 0.956, 0.621],
                               [1.0, -0.272, -0.647],
                               [1.0, -1.106, 1.703]
                             ])
        
        #img_copy[:,:,0] = arr_img[:,:,0] * 1.0 + arr_img[:,:,1] * 0.956 + arr_img[:,:,2] * 0.621
        #img_copy[:,:,1] = arr_img[:,:,0] * 1.0 + arr_img[:,:,1] * -0.272 + arr_img[:,:,2] * -0.647
        #img_copy[:,:,2] = arr_img[:,:,0] * 1.0 + arr_img[:,:,1] * -1.106 + arr_img[:,:,2] * 1.703
        
        img_copy = np.dot(arr_img, matrix_rgb.T.copy())
        
        # Estabelecendo os limites de RGB
        np.where(img_copy < 0, img_copy, 0)
        np.where(img_copy > 255, img_copy, 255)
        
        # Transformando array em imagem
        img_transformed = Image.fromarray(img_copy.astype('uint8'))
        self.image_transformed = Image.fromarray(img_copy.astype('uint8'))
        
        # Retorna a imagem processada
        return img_transformed, img_copy