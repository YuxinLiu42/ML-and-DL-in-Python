# 1. Computer Vision
# 2. Images in Python
# 3. Computer Vision with Python - OpenCV vs Pillow
# 4. OpenCV - First Steps
import cv2
import matplotlib.pyplot as plt

# read an image
img = cv2.imread("training_imgs/flowers.jpg")

if img is None:
    print("Error: Could not read image.")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# Alternative 1 (only display; read has been done already):
import cv2

# Assume img is already read
cv2.imshow("image", img)
cv2.waitKey()  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window displaying the image

# Alternative 2 (read and display):
from PIL import Image

# Read and display using PIL
im = Image.open("training_imgs/flowers.jpg")
im.show()

# Read and display using OpenCV
img = cv2.imread("training_imgs/flowers.jpg")

# Output the object type
print(type(img))

# Output the data type
print(img.dtype)

# Output the array dimensions
print(img.shape)

# Display using OpenCV (Assuming img is already read)
cv2.imshow("image", img)
cv2.waitKey()  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window displaying the image

# Read and display using PIL
im = Image.open("training_imgs/flowers.jpg")
im.show()

# Read using OpenCV
img = cv2.imread("training_imgs/flowers.jpg")

# Output the object type
print("Object type:", type(img))

# Output the data type
print("Data type:", img.dtype)

# Output the array dimensions
print("Array dimensions:", img.shape)

# Output the array size (number of elements in the array)
print("Array size:", img.size)

# Make a copy of the image
img_copy = img.copy()

# Save the copied image
cv2.imwrite("training_imgs/flowers_copy.jpg", img_copy)

# 5. Make Minor Changes to Images
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Display using OpenCV (Assuming img is already read)
cv2.imshow("image", img)
cv2.waitKey()  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window displaying the image

# Read and display using PIL
im = Image.open("training_imgs/flowers.jpg")
im.show()

# Read using OpenCV
img = cv2.imread("training_imgs/flowers.jpg")

# Output the object type
print("Object type:", type(img))

# Output the data type
print("Data type:", img.dtype)

# Output the array dimensions
print("Array dimensions:", img.shape)

# Output the array size (number of elements in the array)
print("Array size:", img.size)

# Accessing individual pixels
# Output the color value of a pixel
pixel = img[0, 0]
print("Pixel at (0,0):", pixel)

# Take only one channel of a pixel
pixel_blue = img[50, 50, 0]
pixel_green = img[50, 50, 1]
pixel_red = img[50, 50, 2]
print("Pixel at (50,50) - Blue:", pixel_blue, "Green:", pixel_green, "Red:", pixel_red)

# Change the color value of a pixel
img_color = img.copy()
img_color[10, 10] = [255, 255, 255]

# Creating image sections
# Cut sections of an image
# Attention: the pixels of a section must be in the image.
# x from to, y from to (first height, then width)
img_part = img[200:300, 100:200]
plt.imshow(cv2.cvtColor(img_part, cv2.COLOR_BGR2RGB))
plt.title("Image Section")
plt.show()

# Changing image size
# Change an image size (number of pixels): smaller
img_resize_small = cv2.resize(img, (50, 50))
plt.imshow(cv2.cvtColor(img_resize_small, cv2.COLOR_BGR2RGB))
plt.title("Resized Image - Smaller")
plt.show()

# Change an image size (number of pixels): larger
img_resize_big = cv2.resize(img, (1000, 1000))
plt.imshow(cv2.cvtColor(img_resize_big, cv2.COLOR_BGR2RGB))
plt.title("Resized Image - Larger")
plt.show()

# Paint on the image: rectangle
img_to_draw_on_rectangle = img.copy()
cv2.rectangle(img_to_draw_on_rectangle,
              pt1=(370, 120),
              pt2=(450, 230),
              color=(0, 0, 255),
              thickness=5)
plt.imshow(cv2.cvtColor(img_to_draw_on_rectangle, cv2.COLOR_BGR2RGB))
plt.title("Rectangle on Image")
plt.show()

# Paint on the image: circle
img_to_draw_on_circle = img.copy()
cv2.circle(img_to_draw_on_circle,
           center=(200, 100),
           radius=30,
           color=(0, 0, 0),
           thickness=10)
plt.imshow(cv2.cvtColor(img_to_draw_on_circle, cv2.COLOR_BGR2RGB))
plt.title("Circle on Image")
plt.show()

# Write on the image
img_to_write_on = img.copy()
cv2.putText(img_to_write_on,
            text='I love python',
            org=(250, 200),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=4,
            lineType=cv2.LINE_AA)
plt.imshow(cv2.cvtColor(img_to_write_on, cv2.COLOR_BGR2RGB))
plt.title("Text on Image")
plt.show()

# 6. Colors in Images
# convert from BGR to RGB
img_BGR2RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_BGR2RGB)

# from RBG to black and white
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_grey, cmap = 'gray')

# 7. Changing Images
# Rotation
(h, w) = img.shape[:2]
center = (w / 2, h / 2)
angle = 46
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)
img_rotated = cv2.warpAffine(img, M, (w, h))
plt.imshow(img_rotated)
plt.title('Rotated Image')
plt.show()

# Reflection: horizontal
img_flip0 = cv2.flip(img, flipCode=0)
plt.imshow(img_flip0)
plt.title('Horizontally Flipped Image')
plt.show()

# Reflection: vertical
img_flip1 = cv2.flip(img, flipCode=1)
plt.imshow(img_flip1)
plt.title('Vertically Flipped Image')
plt.show()

# Blurring: Kernel 7
kernel = 7
img_blurr7 = cv2.blur(img, ksize=(kernel, kernel))
plt.imshow(img_blurr7)
plt.title('Blurred Image with Kernel 7')
plt.show()

# Blurring: Kernel 11
kernel = 11
img_blurr11 = cv2.blur(img, ksize=(kernel, kernel))
plt.imshow(img_blurr11)
plt.title('Blurred Image with Kernel 11')
plt.show()

# Median Blurring
img_median_blur = cv2.medianBlur(img, ksize=7)
plt.imshow(img_median_blur)
plt.title('Median Blurred Image')
plt.show()

# Gaussian filtering
img_gaussian_blur = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=1.5)
plt.imshow(img_gaussian_blur)
plt.title('Gaussian Blurred Image')
plt.show()

# Bilateral Filtering
img_bilateral_filter = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
plt.imshow(img_bilateral_filter)
plt.title('Bilateral Filtered Image')
plt.show()

# Different kernels
# Erosion
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
img_eroded = cv2.erode(img, kernel, iterations=1)
plt.imshow(img_eroded)
plt.title('Eroded Image')
plt.show()

# Dilation
img_dilated = cv2.dilate(img, kernel, iterations=1)
plt.imshow(img_dilated)
plt.title('Dilated Image')
plt.show()



'''
Course: Machine Learning and Deep Learning with Python
SoSe 2024
LMU Munich, Department of Statistics
Exercise 7: Computer Vision Basics
'''

# pip install opencv-python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import pathlib

pathlib.Path('models').mkdir(parents=True, exist_ok=True)
pathlib.Path('plots').mkdir(parents=True, exist_ok=True)

###############################################################################
# If you obtain the error message:
# QObject::moveToThread: Current thread (0x31a16f0) is not the object's thread (0x2eaf780).
# Cannot move to target thread (0x31a16f0)
#
# qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/feurerm/sync_dir/teaching/2023_SoSe_Python/miniconda/envs/lecturepython/lib/python3.11/site-packages/cv2/qt/plugins" even though it was found.
# This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
#
# Please try one the following two alternative backends for displaying figures
# in Python:
# matplotlib.use("TKAgg")
# This require TKInter
# matplotlib.use("WebAgg")
# This requires the package `tornado`
#
###############################################################################

# %% ------------------------------------------------------------------------------------
# BLOCK 1: Reading and Displaying Images
# --------------------------------------
print('#' * 50)
print('########## Reading and Displaying Images ##########')
print('#' * 50)

# %% ------------------------------------------------------------------------------------
# EX01: Read the image 'data/flowers.jpg' using cv2.imread()
print('---------- EX-01 ----------')

img = cv2.imread('training_imgs/flowers.jpg')

# %% ------------------------------------------------------------------------------------
# EX02: Show the image in the console. To this end, use plt.imshow()
print('---------- EX-02 ----------')

plt.imshow(img)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX03: How is the image represented in Python?
# Output the object type and the data type of the image
print('---------- EX-03 ----------')

img_type = type(img)
img_dtype = img.dtype

# It is a NumPy array with integers => we can use everything we learned in the NumPy lecture!

# %% ------------------------------------------------------------------------------------
# EX04: Output the image dimensions
print('---------- EX-04 ----------')

img_shape = img.shape
print(img_shape)

# %% ------------------------------------------------------------------------------------
# EX05: Output the size of the image's array
print('---------- EX-05 ----------')

img_size = img.size
print(img_size)

# %% ------------------------------------------------------------------------------------
# EX06: Make a copy of the image
print('---------- EX-06 ----------')

img_copy = img.copy()
print(img_copy)

# %% ------------------------------------------------------------------------------------
# BLOCK 2: Small Changes on Images
# --------------------------------
print('#' * 50)
print('########## Small Changes on Images ##########')
print('#' * 50)

# %% ------------------------------------------------------------------------------------
# EX01: Output the three color values of an arbitrary pixel
print('---------- EX-01 ----------')

pixel = img[50, 50]
print(pixel)

# %% ------------------------------------------------------------------------------------
# EX02: Output the blue value of the same pixel
# Note: cv2 shows BGR by default
print('---------- EX-02 ----------')

pixel_blue = img[50, 50, 0]
print(pixel_blue)

# %% ------------------------------------------------------------------------------------
# EX03: Change the color values of an arbitrary pixel and output the new values
# You can view the modified image
# Make sure to not change the original image!
# Note: color values are whole numbers from 0 up to and including 255
# (8 bit = integer). (0,0,0) is black, (255, 255, 255) is white.
print('---------- EX-03 ----------')

img_color = img.copy()
img_color[100, 100] = [255, 255, 255]
img_color[100, 100]
plt.imshow(img_color)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX04: Cut out part of the image and display the partition in the console
print('---------- EX-04 ----------')

part = img[273:333, 100:160]
plt.imshow(part)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX05: Change the size (i.e., the number of pixels) of the image with cv2.resize()
# and display the resized image in the console
print('---------- EX-05 ----------')

resize = cv2.resize(img, (500, 500))
plt.imshow(resize)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX06: Writing / Painting something on the image:
# Use cv2.rectangle() to draw a rectangle on the image in an arbitrary color.
# Note: The rectangle should march the image dimensions
# Make sure to not change the original image!
print('---------- EX-06 ----------')

img_to_draw_on = img.copy()
cv2.rectangle(img_to_draw_on,
              pt1=(370, 120),
              pt2=(450, 230),
              color=(255, 0, 0),
              thickness=5)
plt.imshow(img_to_draw_on)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX07: Draw a circle on the image with cv2.circle().
# Note: The circle should match the image dimensions
print('---------- EX-07 ----------')

cv2.circle(img_to_draw_on,
           center=(200, 100),
           radius=30,
           color=(0, 255, 0),
           thickness=5)
plt.imshow(img_to_draw_on)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX08: Write a text on the image with cv2.putText() in an arbitrary color.
print('---------- EX-08 ----------')

cv2.putText(img_to_draw_on,
            text='I love python',
            org=(50, 200),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=(0, 255, 255),
            thickness=4,
            lineType=cv2.LINE_AA)
plt.imshow(img_to_draw_on)
plt.show()

# %% ------------------------------------------------------------------------------------
# BLOCK 3: Color Spaces
# ---------------------
print('#' * 50)
print('########## Color Spaces ##########')
print('#' * 50)

# %% ------------------------------------------------------------------------------------
# EX01: Change the image from BGR to RGB and check the result
print('---------- EX-01 ----------')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX02: Change the image to black and white and check the result
print('---------- EX-02 ----------')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap='gray')
plt.show()

# %% ------------------------------------------------------------------------------------
# BLOCK 4: Flipping, Rotating, and Blurring
# -----------------------------------------
print('#' * 50)
print('########## Flipping, Rotating, and Blurring ##########')
print('#' * 50)

# %% ------------------------------------------------------------------------------------
# EX01: # Rotate the image 46° counterclockwise and output the result
# Note: Helps for the functions can be found at:
#   https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
#   https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#
#   https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=blur#
print('---------- EX-01 ----------')

(h, w) = img.shape[:2]
center = (h / 2, w / 2)
angle = 46
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(img, M, (w, h))
plt.imshow(rotated)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX02: Flip the image vertically with cv2.flip() and output the result
print('---------- EX-02 ----------')

flipped_img = cv2.flip(img, flipCode=1)
plt.imshow(flipped_img)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX03: Flip the image horizontally with cv2.flip() and output the result
print('---------- EX-03 ----------')

flipped_img = cv2.flip(img, flipCode=0)
plt.imshow(flipped_img)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX04: Use cv2.blur() to blur the image and output the result
# Try several kernel sizes.
print('---------- EX-04 ----------')

kernel = 7
img_blurred_7 = cv2.blur(img, ksize=(kernel, kernel))
plt.imshow(img_blurred_7)
plt.show()

kernel = 11
img_blurred_11 = cv2.blur(img, ksize=(kernel, kernel))
plt.imshow(img_blurred_11)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX05: Blur the image by applying the functions:
# - cv2.GaussianBlur()
# - cv2.medianBlur()
# - cv2.bilateralFilter()
print('---------- EX-05 ----------')

img_gaus_blur = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=0)
plt.imshow(img_gaus_blur)
plt.show()
img_med_blur = cv2.medianBlur(img, 7)
plt.imshow(img_med_blur)
plt.show()
img_bil_filter = cv2.bilateralFilter(img, 7, sigmaSpace=75, sigmaColor=75)
plt.imshow(img_bil_filter)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX06: Apply erosion kernels to the image with cv2.erode() and check the result.
# Try several kernels.
print('---------- EX-06 ----------')

kernel = np.ones((4, 4), np.uint8)
img_errosion = cv2.erode(img, kernel, iterations=3)
plt.imshow(img_errosion)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX07: Apply dilation kernels to the image and and output the result.
# Try several kernels and cv2.dilate()
print('---------- EX-07 ----------')

kernel = np.ones((2, 2), np.uint8)
img_dilate = cv2.dilate(img, kernel, iterations=3)
plt.imshow(img_dilate)
plt.show()

# %% ------------------------------------------------------------------------------------
# EX08: Write a function that automatically augments images.
# The function should return the augmented image.
# The change method should be randomly selected for each function call.
print('---------- EX-08 ----------')


def get_aug_img(img):
    img_aug = [cv2.blur(img, ksize=(11, 11)),  # Plot the image with different kernel sizes
               cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=0),  # Blur the image
               cv2.erode(img, (4, 4), iterations=3),  # Create erosion kernels
               cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=3),  # Apply dilation
               cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
               cv2.cvtColor(img, cv2.COLOR_BGR2HLS),
               cv2.flip(img, flipCode=1),
               cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[:2][0] / 2, img.shape[:2][1] / 2), 80, 1),
                              (img.shape[:2][0], img.shape[:2][1]))]
    return img_aug[np.random.randint(8)]


img = cv2.imread('training_imgs/flowers.jpg')
img_aug = get_aug_img(img)
plt.imshow(img_aug)
plt.show()

# %% ------------------------------------------------------------------------------------
# BLOCK 5: Saving Images
# ----------------------
print('#' * 50)
print('########## Saving Images ##########')
print('#' * 50)

# %% ------------------------------------------------------------------------------------
# EX01: Save one of the newly created images with cv2.imwrite() in the data folder.
# Help for this:
# https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#
print('---------- EX-01 ----------')

cv2.imwrite('data/img_aug.jpg', img_aug)
