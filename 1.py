import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color

def detect_coins():

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


def segment_and_count_coins():
  # Load the image
  image = cv2.imread('./images/sample.png')
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply Gaussian blur to reduce noise
  blurred = cv2.GaussianBlur(gray, (15, 15), 0)

  # Thresholding for segmentation
  _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # Apply morphological operations to remove small noise
  kernel = np.ones((15, 15), np.uint8)
  cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

  # Identify connected components
  labels = measure.label(cleaned, connectivity=2)
  colored_labels = color.label2rgb(labels, bg_label=0)

  # Count coins by counting unique labels (excluding background)
  num_coins = len(np.unique(labels)) - 1
  print(f"Number of coins detected: {num_coins}")

  # Display segmented coins
  plt.figure(figsize=(10, 6))
  plt.subplot(1, 2, 1)
  plt.title("Original Image")
  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  plt.subplot(1, 2, 2)
  plt.title("Segmented Coins")
  plt.imshow(colored_labels)
  plt.show()


detect_coins()

segment_and_count_coins()
