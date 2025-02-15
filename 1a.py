import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('./images/sample.png', cv2.IMREAD_GRAYSCALE)

# Gaussian Blur
blurred = cv2.GaussianBlur(image, (5, 5), 1.5)

# Canny Edge Detection
edges = cv2.Canny(blurred, threshold1=70, threshold2=140)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()