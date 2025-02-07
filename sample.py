import cv2 as cv
import graphicshelpers as gh


class Sample:
    def __init__(self, path, label):
        self.path = path
        self.label = label
        self.image = gh.read_image(path)
        self.image_filtered = gh.prepare_image(self.image)
        self.kp, self.desc, self.image_kp = self.prepare_desc()

    def get_image(self):
        return self.image

    def get_image_filtered(self):
        return self.image_filtered

    def get_kp(self):
        return self.kp

    def get_desc(self):
        return self.desc

    def get_image_kp(self):
        return self.image_kp

    def prepare_desc(self):
        kp, desc = gh.extract_fetures(self.image_filtered)
        image_kp = cv.drawKeypoints(self.image_filtered, kp, 0)
        return kp, desc, image_kp

