import numpy as np
import cv2
import glob
import imutils


class ImageKeypoint:
    def __init__(self, im, ):
        self.img = im
        self.sift_detector = cv2.xfeatures2d.SIFT_create()
        key_points, descriptors = self.sift_detector.detectAndCompute(img, None)
        self.key_pts = key_points
        self.descr = descriptors

    def image(self):
        return self.img

    def key_points(self):
        return self.key_pts

    def descriptors(self):
        return self.descr

    def key_points_and_detectors(self):
        return self.key_pts, self.descr

def stich_images_own():

    image_paths = glob.glob('../images/*.jpg')
    images = []
    for image in image_paths:
        img = cv2.imread(image)
        images.append(img)
        cv2.imshow("Images", img)
        cv2.waitKey(0)

    img_keypoint_descriptors = []
    for img in images:
        img_keypoint_descriptors.append(ImageKeypoint(img))

    # create feature matcher
    # pair two key_descr pair to one pair : [[k1,d1],[k2,d2],[k3,d3],[k4,d4]] -> [[[k1,d1],[k2,d2]], [[k3,d3],[k4,d4]]]
    # then match both descriptors
    # key_descr_pair = zip(key_descr[0::2], key_descr[1::2])
    matched_images = []
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    for index, ikd_1 in enumerate(img_keypoint_descriptors):
        # i_k_d = image_keypoint_descriptor object
        for ikd_2 in img_keypoint_descriptors[:index]:
            all_matches = bf.match(ikd_1.descriptors(), ikd_2.descriptors())
            # sort matches by distance
            sorted_matches = sorted(all_matches, key=lambda x: x.distance)
            matched_images.append(cv2.drawMatches(ikd_1.image(), ikd_1.key_points(), ikd_2.image(), ikd_2.key_points(), sorted_matches[:5], ikd_2.image(), flags=2))

    # show the image
    i = 0
    for matched_img in matched_images:
        cv2.imshow('image', matched_img)
        # save the image
        cv2.imwrite(f"matched_images{i}.jpg", matched_img)
        i = i + 1
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def stitch_images():
    image_paths = glob.glob('../images/*.jpg')
    images = []


    for image in image_paths:
        img = cv2.imread(image)
        images.append(img)
        cv2.imshow("Image", img)
        cv2.waitKey(0)

    imageStitcher = cv2.Stitcher_create()

    error, stitched_img = imageStitcher.stitch(images)

    if not error:

        cv2.imwrite("stitchedOutput.png", stitched_img)
        cv2.imshow("Stitched Img", stitched_img)
        cv2.waitKey(0)




        # stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
        #
        # gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        # thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]
        #
        # cv2.imshow("Threshold Image", thresh_img)
        # cv2.waitKey(0)
        #
        # contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # contours = imutils.grab_contours(contours)
        # areaOI = max(contours, key=cv2.contourArea)
        #
        # mask = np.zeros(thresh_img.shape, dtype="uint8")
        # x, y, w, h = cv2.boundingRect(areaOI)
        # cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)
        #
        # minRectangle = mask.copy()
        # sub = mask.copy()
        #
        # while cv2.countNonZero(sub) > 0:
        #     minRectangle = cv2.erode(minRectangle, None)
        #     sub = cv2.subtract(minRectangle, thresh_img)
        #
        #
        # contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # contours = imutils.grab_contours(contours)
        # areaOI = max(contours, key=cv2.contourArea)
        #
        # cv2.imshow("minRectangle Image", minRectangle)
        # cv2.waitKey(0)
        #
        # x, y, w, h = cv2.boundingRect(areaOI)
        #
        # stitched_img = stitched_img[y:y + h, x:x + w]
        #
        # cv2.imwrite("stitchedOutputProcessed.png", stitched_img)
        #
        # cv2.imshow("Stitched Image Processed", stitched_img)
        #
        # cv2.waitKey(0)



    else:
        print("Images could not be stitched!")
        print("Likely not enough keypoints being detected!")


if __name__=="__main__":
    stitch_images()