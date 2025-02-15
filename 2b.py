import cv2
import numpy as np

def load_images(image1_path, image2_path):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    return img1, img2

def detect_and_match_features(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    return keypoints1, keypoints2, matches

def draw_matches(img1, img2, keypoints1, keypoints2, matches):
    # Draw matches
    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image

def stitch_images(img1, img2, keypoints1, keypoints2, matches):
    # Extract location of matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the first image onto the second
    height2, width2 = img2.shape[:2]
    height1, width1 = img1.shape[:2]

    # Calculate the size of the panorama
    panorama_width = width1 + width2
    panorama_height = max(height1, height2)

    # Warp the first image
    warped_img1 = cv2.warpPerspective(img1, H, (panorama_width, panorama_height))

    # Place the second image on the panorama
    warped_img1[0:height2, 0:width2] = img2

    return warped_img1

def main():
    # Load images
    img1, img2 = load_images('IMG_4474.png','IMG_4473.png')

    # Detect and match features
    keypoints1, keypoints2, matches = detect_and_match_features(img1, img2)

    # Draw matches
    matched_image = draw_matches(img1, img2, keypoints1, keypoints2, matches)
    cv2.imshow('Matched Features', matched_image)

    # Stitch images into panorama
    panorama = stitch_images(img1, img2, keypoints1, keypoints2, matches)
    cv2.imshow('Panorama', panorama)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
