import matplotlib.pyplot as plt
from main import *

image_path_1 = r"images\bird.jpg"
image_path_2 = r"images\building.jpg"
image_path_3 = r"images\lemons.jpg"
image_path_4 = r"images\stork.jpg"

img_bird = load_image(image_path_1)
img_building = load_image(image_path_2)
img_lemons = load_image(image_path_3)
img_stork = load_image(image_path_4)

#GRAY-SCALE IMAGE
"""
gray_image = convert_to_gray(img_building)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img_building)
plt.title("Original Image")
plt.subplot(1,2,2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.show()

"""

#BINARY IMAGE
"""
gray_image = convert_to_gray(image_array=img_lemons)
binary_img = binary_threshold(image_array=gray_image, threshold=100.0)

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_lemons)
plt.subplot(1,2,2)
plt.title("Binary Image")
plt.imshow(binary_img)
plt.show()


"""

#IMAGE ROTATE
"""
gray_image = convert_to_gray(img_bird)
rotated_img = rotate_image(image_array=gray_image, degree=30)

plt.figure(figsize=(10,6))
plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img_bird)
plt.subplot(1,3,2)
plt.title("Grayscale Image")
plt.imshow(gray_image, cmap="gray")
plt.subplot(1,3,3)
plt.title("Rotated Image")
plt.imshow(rotated_img, cmap="gray")
plt.show()

"""

#CROPPED IMAGE
"""
cropped_img = crop_image(image_array=img_building, x=2000, y=3000, w=2000, h=3000)

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_building)
plt.subplot(1,2,2)
plt.title("Cropped Image")
plt.imshow(cropped_img, cmap="gray")
plt.show()

"""

#COLOR - SPACE CONVERSION
"""
Y, I, Q = rgb_to_yiq(img_stork)

plt.figure(figsize=(15,4))
plt.subplot(1,4,1)
plt.title("Original Image")
plt.imshow(img_stork)
plt.subplot(1,4,2)
plt.title("Y: Parlaklık")
plt.imshow(Y)
plt.subplot(1,4,3)
plt.title("I: Ton, Adı")
plt.imshow(I)
plt.subplot(1,4,4)
plt.title("Q: Saturasyon, Doygunluk")
plt.imshow(Q)
plt.show()

"""

#HISTOGRAM STRECTING
"""
gray_image = convert_to_gray(img_building)
hist_strec_img = histogram_strecting(image_array=gray_image, max_value=600, min_value=50)

plt.figure(figsize=(10,6))
plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img_building)
plt.subplot(1,3,2)
plt.title("Grayscale Image")
plt.imshow(gray_image, cmap="gray")
plt.subplot(1,3,3)
plt.title("Histogram Stretcting")
plt.imshow(hist_strec_img, cmap="gray")
plt.show()

"""

#ARITHMETIC OPERATIONS BETWEEN TWO IMAGES
#ADDITIONS AND MULTIPLY
"""
gray_image1 = convert_to_gray(img_lemons)
gray_image2 = convert_to_gray(img_stork)

h = min(gray_image1.shape[0], gray_image2.shape[0])
w = min(gray_image1.shape[1], gray_image2.shape[1])

gray_image1 = gray_image1[ :h,  :w]
gray_image2 = gray_image2[ :h,  :w]

added    = add_images(gray_image1, gray_image2)
multiplied = multiply_images(gray_image1, gray_image2)

plt.figure(figsize=(15,4))
plt.subplot(1,4,1)
plt.title("Original Image-1")
plt.imshow(gray_image1, cmap="gray")
plt.subplot(1,4,2)
plt.title("Original Image-2")
plt.imshow(gray_image2, cmap="gray")
plt.subplot(1,4,3)
plt.title("Added of Two Images")
plt.imshow(added, cmap="gray")
plt.subplot(1,4,4)
plt.title("Multiplied of Two Images")
plt.imshow(multiplied, cmap="gray")
plt.show()

"""

#INCREASE BRIGHTNESS
"""
brighter_img = adjust_brightness(image=img_bird, beta=100)

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_bird)
plt.subplot(1,2,2)
plt.title("Brighter Image")
plt.imshow(brighter_img)
plt.show()

"""
#OPERATION OF CONVOLUTION WITH GAUSSIAN KERNEL
"""
gray_image = convert_to_gray(img_lemons)
gaussian_img = gaussian_blur(image=gray_image, kernel_size=7, sigma=2)

plt.figure(figsize=(10,6))
plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img_lemons)
plt.subplot(1,3,2)
plt.title("Grayscale Image")
plt.imshow(gray_image, cmap="gray")
plt.subplot(1,3,3)
plt.title("Gaussian Blur Image")
plt.imshow(gaussian_img, cmap="gray")
plt.show()

"""

#ADAPTIVE THRESHOLDING
"""
gray_image = convert_to_gray(img_bird)
threshold_img = adaptive_threshold(image=gray_image, kernel_size=7)

plt.figure(figsize=(10,6))
plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img_bird)
plt.subplot(1,3,2)
plt.title("Grayscale Image")
plt.imshow(gray_image, cmap="gray")
plt.subplot(1,3,3)
plt.title("Threholding Image")
plt.imshow(threshold_img, cmap="gray")
plt.show()

"""
#EDGE DETECTION WITH SOBEL METHOD
"""
gray_image = convert_to_gray(img_building)
gaussian_img = gaussian_blur(image=gray_image, kernel_size=11, sigma=2)
sobel_img = sobel_edge_detection(gaussian_img)

plt.figure(figsize=(10,6))
plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img_building)
plt.subplot(1,3,2)
plt.title("Grayscale Image")
plt.imshow(gray_image, cmap="gray")
plt.subplot(1,3,3)
plt.title("Edge Detected with Sobel Image")
plt.imshow(sobel_img, cmap="gray")
plt.show()

"""

#TO ADD SOME NOISE TO THE IMAGE: SALT & PEPPER
"""
gray_image = convert_to_gray(img_bird)
salt_pepper_img = add_to_noise(image=gray_image, salt_ratio=0.2, pepper_ratio=0.2)

plt.figure(figsize=(10,6))
plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img_bird)
plt.subplot(1,3,2)
plt.title("Grayscale Image")
plt.imshow(gray_image, cmap="gray")
plt.subplot(1,3,3)
plt.title("Salt&Pepper Noise Image")
plt.imshow(salt_pepper_img, cmap="gray")
plt.show()

"""

#TO CLEAN THE SALT&PEPPER NOISE WITH MEDIAN AND MEAN FILTER
"""
gray_image = convert_to_gray(img_bird)
salt_pepper_img = add_to_noise(image=gray_image, salt_ratio=0.2, pepper_ratio=0.2)

mean_filter = clear_noise_filter_with_mean(image=salt_pepper_img, kernel_size=3)
median_filter = clear_noise_filter_with_median(image=salt_pepper_img, kernel_size=3)

plt.figure(figsize=(10,6))
plt.subplot(1,3,1)
plt.title("Salt&Pepper Noise Image")
plt.imshow(salt_pepper_img, cmap="gray")
plt.subplot(1,3,2)
plt.title("Mean Filter")
plt.imshow(mean_filter, cmap="gray")
plt.subplot(1,3,3)
plt.title("Median Filter")
plt.imshow(median_filter, cmap="gray")
plt.show()

"""
#MORPHOLOGY PROCESS: DILATE, EROSION, OPEN, CLOSE
"""
gray_image = convert_to_gray(img_stork)

se = create_structuring_element(size=5)
dilated_img = dilate(gray_image, se)
erode_img = erode(gray_image, se)
close_img = erode_dilate_close(gray_image, se)
open_img = dilate_erode_open(gray_image, se)

plt.figure(figsize=(15,6))
plt.subplot(1,5,1)
plt.title("Original Image")
plt.imshow(gray_image, cmap="gray")
plt.subplot(1,5,2)
plt.title("Dilated Image")
plt.imshow(dilated_img, cmap="gray")
plt.subplot(1,5,3)
plt.title("Eroded Image")
plt.imshow(erode_img, cmap="gray")
plt.subplot(1,5,4)
plt.title("Dilate->Erode Image")
plt.imshow(close_img, cmap="gray")
plt.subplot(1,5,5)
plt.title("Erode->Dilate Image")
plt.imshow(open_img, cmap="gray")
plt.show()

"""


