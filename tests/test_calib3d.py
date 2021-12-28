import cv2
from math import floor
import numpy as np
import unittest

from ga_opencv_python import calib3d


# TODO: Have input be test fixture
# TODO: Round input
class TestCameraCalibrationAnd3DReconstruction(unittest.TestCase):
    def test_projectPoints(self):
        pointcloud = np.array([[ 1.1793237 ,  0.13069004, -0.04256555]], dtype=np.float32)
        distortion_coeffs= np.array([0.03797018, -0.10685437, 0.02071762, -0.01338533, 0.2683136], dtype=np.float64)
        camera_matrix = np.array([[1.83771013e+03, 0.00000000e+00, 1.12993096e+03],
                                  [0.00000000e+00, 1.83920740e+03, 1.09961561e+03],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
        translation = np.array([[0.04037025], [-1.58508158], [2.46504903]], dtype=np.float64)
        rotation = np.array([[-0.99877697,  0.0190814 , -0.04561208],
                             [ 0.01525185,  0.99644283,  0.08287985],
                             [ 0.04703129,  0.08208281, -0.99551518]], dtype=np.float64)
        # Compare GPU results against actual OpenCV results
        # TODO: Currently experiencing small differences, presumably due to floating point precision?
        res1, _ = calib3d.camera_calibration_and_3d_reconstruction.projectPoints(pointcloud,
                                                                                 rotation,
                                                                                 translation,
                                                                                 camera_matrix,
                                                                                 distortion_coeffs)
        res2, _ = cv2.projectPoints(pointcloud,
                                    rotation,
                                    translation,
                                    camera_matrix,
                                    distortion_coeffs)
        assert np.allclose(res1, res2, rtol=1e-2)

    def test_projectPoints_no_distortion(self):
        pointcloud = np.array([[ 1.1793237 ,  0.13069004, -0.04256555]], dtype=np.float32)
        camera_matrix = np.array([[1.83771013e+03, 0.00000000e+00, 1.12993096e+03],
                                  [0.00000000e+00, 1.83920740e+03, 1.09961561e+03],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
        translation = np.array([[0.04037025], [-1.58508158], [2.46504903]], dtype=np.float64)
        rotation = np.array([[-0.99877697,  0.0190814 , -0.04561208],
                             [ 0.01525185,  0.99644283,  0.08287985],
                             [ 0.04703129,  0.08208281, -0.99551518]], dtype=np.float64)
        # Compare GPU results against actual OpenCV results
        # TODO: Currently experiencing small differences, presumably due to floating point precision?
        res1, _ = calib3d.camera_calibration_and_3d_reconstruction.projectPoints(pointcloud,
                                                                                 rotation,
                                                                                 translation,
                                                                                 camera_matrix,
                                                                                 None)
        res2, _ = cv2.projectPoints(pointcloud,
                                    rotation,
                                    translation,
                                    camera_matrix,
                                    None)
        assert np.allclose(res1, res2, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
