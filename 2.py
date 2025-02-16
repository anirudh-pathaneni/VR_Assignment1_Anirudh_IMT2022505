import cv2
import numpy as np

def load_images(image1_path, image2_path):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    return img1, img2

def detect_keypoints(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints (without computing matches)
    keypoints1 = sift.detect(img1, None)
    keypoints2 = sift.detect(img2, None)

    return keypoints1, keypoints2

def draw_keypoints(img1, img2, keypoints1, keypoints2):
    img1_with_keypoints = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_with_keypoints = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img1_with_keypoints, img2_with_keypoints

def stitch_images(img1, img2, keypoints1, keypoints2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.compute(img1, keypoints1)
    keypoints2, descriptors2 = sift.compute(img2, keypoints2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    height2, width2 = img2.shape[:2]
    height1, width1 = img1.shape[:2]
    panorama_width = width1 + width2
    panorama_height = max(height1, height2)

    warped_img1 = cv2.warpPerspective(img1, H, (panorama_width, panorama_height))
    warped_img1[0:height2, 0:width2] = img2

    return warped_img1


# Load images
img1, img2 = load_images('./images/IMG_4474.png','./images/IMG_4473.png')

# Detect keypoints
keypoints1, keypoints2 = detect_keypoints(img1, img2)

# Draw keypoints
img1_with_keypoints, img2_with_keypoints = draw_keypoints(img1, img2, keypoints1, keypoints2)

cv2.imshow('Image 1 Keypoints', img1_with_keypoints)
cv2.imshow('Image 2 Keypoints', img2_with_keypoints)

# Stitch images
panorama = stitch_images(img1, img2, keypoints1, keypoints2)
cv2.imshow('Panorama', panorama)

cv2.waitKey(0)
cv2.destroyAllWindows()
