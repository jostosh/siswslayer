from face_frontalization import frontalize
from face_frontalization import facial_feature_detector as feature_detection
from face_frontalization import camera_calibration as calib
import scipy.io as io
import numpy as np


class Frontalizer:

    def __init__(self):
        self.model3D = frontalize.ThreeD_Model(
            "/home/jos/siswslayer/face_frontalization/frontalization_models/model3Ddlib.mat", 'model_dlib'
        )
        self.eyemask = np.asarray(io.loadmat('/home/jos/siswslayer/face_frontalization/frontalization_models/eyemask.mat')['eyemask'])

    def transform(self, img):
        lmarks = feature_detection.get_landmarks(img)
        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(self.model3D, lmarks[0])
        frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, self.model3D.ref_U, self.eyemask)
        return frontal_raw, frontal_sym
