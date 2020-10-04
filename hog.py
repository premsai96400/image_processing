from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt



image = imread("C:\\Users\\Hi\\Pictures\\Camera Roll\\kiran.jpg")
image = resize(image, (400,400))
#image = imread("E:\\asl_data set\\b\\hand1_b_bot_seg_1_cropped.jpeg")

fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
                    

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 6))

ax2.axis('off')
ax2.imshow(hog_image)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
print(fd,len(fd))