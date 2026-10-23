import numpy as np
from PIL import Image

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.array(img)

#RGB -> GRAY-SCALE CONVERSION
def convert_to_gray(image_array):
    gray_scale = (image_array[:, :, 0]*0.299 + image_array[:, :, 1]*0.587 + image_array[:, :, 2]*0.114)
    return gray_scale

#BINARY CONVERSION
def binary_threshold(image_array, threshold):
    height, width = image_array.shape[0], image_array.shape[1]
    binary_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):

            if image_array[i, j] > threshold:
                binary_image[i,j] = 255

            else:
                binary_image[i,j] = 0

    return binary_image

#IMAGE ROTATE
def rotate_image(image_array, degree):

    theta = np.deg2rad(degree)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    height, width = image_array.shape

    # Yeni boyutları hesapla (köşelerin döndürülmüş haline göre)
    new_height = int(abs(height * cos_theta) + abs(width * sin_theta))
    new_width = int(abs(width * cos_theta) + abs(height * sin_theta))

    rotated_image = np.zeros((new_height, new_width), dtype=np.uint8)

    old_center_x, old_center_y = width / 2, height / 2
    new_center_x, new_center_y = new_width / 2, new_height / 2

    for i in range(new_height):
        for j in range(new_width):

            x = (j - new_center_x) * cos_theta + (i - new_center_y) * sin_theta + old_center_x
            y = -(j - new_center_x) * sin_theta + (i - new_center_y) * cos_theta + old_center_y

            # Eğer (x, y) görüntü sınırları içinde ise, bilinear interpolasyon uygula
            if 0 <= x < width - 1 and 0 <= y < height - 1:
                x0 = int(np.floor(x))
                x1 = x0 + 1
                y0 = int(np.floor(y))
                y1 = y0 + 1

                a = x - x0
                b = y - y0

                # 4 komşu pikseli al
                I00 = image_array[y0, x0]
                I01 = image_array[y1, x0]
                I10 = image_array[y0, x1]
                I11 = image_array[y1, x1]

                # Bilinear interpolasyon
                interpolated_value = (1 - a) * (1 - b) * I00 + \
                                     (1 - a) * b * I01 + \
                                     a * (1 - b) * I10 + \
                                     a * b * I11

                rotated_image[i, j] = int(interpolated_value)

    return rotated_image

#CROPPED IMAGE
def crop_image(image_array, x, y, w, h):
    height, width = image_array.shape[0], image_array.shape[1]

    # Kenar taşmalarına karşı sınır kontrolü
    if x + w > width or y + h > height:
        print("Geçersiz Değerler")
    
    if len(image_array.shape) == 3:
        cropped_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                cropped_image[i][j] = image_array[y + i][x + j]
    else:
        cropped_image = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                cropped_image[i][j] = image_array[y + i][x + j]

    return cropped_image

#ZOOM IMAGE
def zoom_image(image_array, scale_factor):
    """
    Görüntüyü yakınlaştırır veya uzaklaştırır.
    scale_factor > 1 → Yakınlaştırma
    scale_factor < 1 → Uzaklaştırma 
    """
    height, width = image_array.shape[0], image_array.shape[1]

    if scale_factor > 1:
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        zoomed_image = np.zeros((new_height, new_width), dtype=np.uint8)

        for i in range(new_height):
            for j in range(new_width):
                src_i = min(height - 1, int(i / scale_factor))
                src_j = min(width - 1, int(j / scale_factor))
                zoomed_image[i,j] = image_array[src_i, src_j]
    
    elif 0 < scale_factor < 1:
        new_height = int(height * scale_factor)
        new_width =  int(width * scale_factor)
        zoomed_image = np.zeros((new_height, new_width), dtype=np.uint8)

        for i in range(new_height):
            for j in range(new_width):
                src_i = int(i / scale_factor)
                src_j = int(j / scale_factor)
                if src_i < height and src_j < width:
                    zoomed_image[i,j] = image_array[src_i, src_j]
    else:
        print("Scale factor 0'dan küçük veya eşit olamaz!")

    return zoomed_image

#COLOR-SPACA CONVERSION
def rgb_to_yiq(image):
    R = image[:, :, 0] / 255.0
    G = image[:, :, 1] / 255.0
    B = image[:, :, 2] / 255.0

    Y =  0.299 * R + 0.587 * G + 0.114 * B
    I =  0.596 * R - 0.274 * G - 0.322 * B
    Q =  0.211 * R - 0.523 * G + 0.312 * B

    return Y, I, Q
############################################
def rgb_to_cmyk(image):
    R = image[:, :, 0] / 255.0
    G = image[:, :, 1] / 255.0
    B = image[:, :, 2] / 255.0

    K = 1 - np.maximum(np.maximum(R, G), B)

    # K == 1 olduğunda (tamamen siyah piksel) bölme hatası oluşur
    # Bu yüzden bölen sıfır olduğunda 0 yaz
    denom = 1 - K
    denom[denom == 0] = 1  # sıfıra bölmeyi engelle

    C = (1 - R - K) / denom
    M = (1 - G - K) / denom
    Y = (1 - B - K) / denom

    return C, M, Y, K

#HISTOGRAM STRECTING
def histogram_strecting(image_array, min_value, max_value):
    """
    Histogram germe (stretching). Görüntüdeki piksel aralığını belirlediğimiz alana yayar.
    Gri görüntü için geçerlidir.
    """
    p_min = image_array.min()
    p_max = image_array.max()

    height, width = image_array.shape[0], image_array.shape[1]
    stretched_image = np.zeros((height, width), dtype=np.uint8)

    image_array = image_array.astype(np.float64) 

    for i in range(height):
        for j in range(width):
            stretched_image[i, j] = (image_array[i, j] - p_min) * (max_value - min_value) / (p_max - p_min) + min_value
    
    return np.clip(stretched_image, 0, 255).astype(np.uint8)

#ARITHMETIC OPERATIONS BETWEEN TWO IMAGES
#OPERATIONS: ADDITION
def add_images(image1, image2):
    # Boyut kontrolü
    if image1.shape != image2.shape:
        raise ValueError("Görüntü boyutları aynı olmalı!")
    
    output = np.clip(image1.astype(np.int32) + image2.astype(np.int32), 0, 255).astype(np.uint8)
    return output
#OPERATIONS: MULTIPLY
def multiply_images(image1, image2):
    # Boyut kontrolü
    if image1.shape != image2.shape:
        raise ValueError("Görüntü boyutları aynı olmalı!")
    
    output = np.clip((image1.astype(np.int32) * image2.astype(np.int32)) / 255, 0, 255).astype(np.uint8)
    return output

#INCREASE BRIGHTNESS
def adjust_brightness(image, beta):
    output = np.clip(image.astype(np.int32) + beta, 0, 255).astype(np.uint8)
    return output

#ADAPTIVE THRESHOLDING
def adaptive_threshold(image, kernel_size, C):
    pad = kernel_size // 2
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)
    padded = np.pad(image, pad, mode='edge')

    for i in range(h):
        for j in range(w):
            # 1. Komşuluk penceresini al
            window = padded[i:i+kernel_size, j:j+kernel_size]
            
            # 2. Pencerenin ortalamasını hesapla
            threshold = np.sum(window) / (kernel_size * kernel_size) - C
            
            # 3. Piksel değerini eşikle karşılaştır
            if image[i, j] > threshold:
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output

#EDGE DETECTION WITH SOBEL
def sobel_edge_detection(image):
    # 1. Sobel Kernelleri
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)

    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)

    pad = 1
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.float64)
    padded = np.pad(image, pad, mode='edge')

    for i in range(h):
        for j in range(w):
            window = padded[i:i+3, j:j+3]

            Gx = np.sum(window * Kx)
            Gy = np.sum(window * Ky)

            output[i, j] = np.sqrt(Gx**2 + Gy**2) #Gradient genlik değeri formülü

    return np.clip(output, 0, 255).astype(np.uint8)

#ADD TO NOISE -> SALT&PEPPER
def add_to_noise(image, salt_ratio, pepper_ratio):
    noisy_img = image.copy()
    total_pixels = image.shape[0] * image.shape[1]

    #SALT PIXELS
    salt = int(total_pixels * salt_ratio)
    salt_rows = np.random.randint(0, image.shape[0], salt)
    salt_cols = np.random.randint(0, image.shape[1], salt)
    noisy_img[salt_rows, salt_cols] = 255

    #PEPPER PIXELS
    pepper = int(total_pixels * pepper_ratio)
    salt_rows = np.random.randint(0, image.shape[0], pepper)
    salt_cols = np.random.randint(0, image.shape[1], pepper)
    noisy_img[salt_rows, salt_cols] = 0


    return noisy_img

#CLEAR NOISE WITH MEAN FILTER
def clear_noise_filter_with_mean(image, kernel_size=3):
    pad = kernel_size // 2
    h, w = image.shape
    output = np.zeros((h,w), dtype=np.uint8)
    padded = np.pad(image, pad, mode="edge")

    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            output[i,j] = np.sum(window) / (kernel_size * kernel_size)

    return output

#CLEAR NOISE WITH MEDIAN FILTER
def clear_noise_filter_with_median(image, kernel_size=3):
    pad = kernel_size // 2
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)
    padded = np.pad(image, pad, mode='edge')

    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            flat = window.flatten()
            flat.sort()
            output[i, j] = flat[len(flat) // 2]

    return output

#ADD BLUR TO IMAGE
def gaussian_kernel(size, sigma):
    kernel = np.zeros((size,size), dtype=np.float64)
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i,j] = np.exp(-(x**2 + y**2) / (2*sigma**2))
        
    kernel /= kernel.sum()  # normalize
    return kernel
########################################
def gaussian_blur(image, kernel_size, sigma):
    pad = kernel_size // 2
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)
    padded = np.pad(image, pad, mode='edge')
    kernel = gaussian_kernel(kernel_size, sigma)  # oluşturduğun kernel

    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(window * kernel)

    return output

#CREATE STRUCTRING ELEMENT TO MORPHOLOGY PROCESS
def create_structuring_element(size):
    return np.ones((size,size), dtype=np.uint8)
#DILATE
def dilate(image, se):
    pad = se.shape[0] // 2
    h, w = image.shape
    output = np.zeros((h,w), dtype=np.uint8)
    padding = np.pad(image, pad, mode="edge")

    for i in range(h):
        for j in range(w):
            window = padding[i:i+se.shape[0], j:j+se.shape[1]]
            output[i, j] = np.max(window[se == 1])
    
    return output
#ERODE
def erode(image, se):
    pad = se.shape[0] // 2
    h, w = image.shape
    output = np.zeros((h,w), dtype=np.uint8)
    padding = np.pad(image, pad, mode="edge")

    for i in range(h):
        for j in range(w):
            window = padding[i:i+se.shape[0], j:j+se.shape[1]]
            output[i, j] = np.min(window[se == 1])
    
    return output
#CLOSE - DILATE->ERODE
def erode_dilate_close(image, se):
    return erode(dilate(image, se), se)
#OPEN - ERODE->DILATE
def dilate_erode_open(image, se):
    return dilate(erode(image, se), se)
##############################################################


