import cv2
import glob

image_paths = glob.glob('../images/*.jpg')
images = []
for image in image_paths:
    img = cv2.imread(image)
    images.append(img)
    cv2.imshow("Images", img)
    cv2.waitKey(0)


# image_stitcher = cv2.Stitcher_create()
# error, stitched_img = image_stitcher.stitch(images)
# if error:
#     print(f"Got error {error}")
#     exit(error)
# cv2.imwrite("stitched_images.jpg", stitched_img)
# cv2.imshow("Stitched Images", stitched_img)
# cv2.waitKey(0)

# detect SIFT features in both images
class ImageKeypoint:
    def __init__(self, im, ):
        self.image = im
        self.sift_detector = cv2.xfeatures2d.SIFT_create()
        key_points, descriptors = self.sift_detector.detectAndCompute(img, None)
        self.key_points = key_points
        self.descriptors = descriptors

    def get_key_points(self):
        return self.key_points

    def get_descriptors(self):
        return self.descriptors

    def get_key_points_and_detectors(self):
        return self.key_points, self.descriptors


img_keypoint_descriptors = []
for img in images:
    img_keypoint_descriptors.append(ImageKeypoint(img))

# create feature matcher
# pair two key_descr pair to one pair : [[k1,d1],[k2,d2],[k3,d3],[k4,d4]] -> [[[k1,d1],[k2,d2]], [[k3,d3],[k4,d4]]]
# then match both descriptors
# key_descr_pair = zip(key_descr[0::2], key_descr[1::2])

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
for index, elem in enumerate(img_keypoint_descriptors):


matches = map((lambda x: bf.match(x[0][1], x[1][1])), key_descr_pair)
# sort matches by distance
sorted_matches = sorted(matches, key=lambda x: x[0].distance)
# draw first 50 matches
img_pair = zip(images[0::2], images[1::2])
matched_imgs = map(lambda x: cv2.drawMatches(x[0][0], x[1][0][0], x[0][1], x[1][1][0], [2][:50], x[0][1], flags=2),
                   zip(img_pair, key_descr_pair, sorted_matches))
# show the image
i = 0
for matched_img in matched_imgs:
    cv2.imshow('image', matched_img)
    # save the image
    cv2.imwrite(f"matched_images{i}.jpg", matched_img)
    i = i + 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
