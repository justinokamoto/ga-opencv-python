import unittest

from ga_opencv_python import calib3d


class TestCameraCalibrationAnd3DReconstruction(unittest.TestCase):
    def test_projectPoints_no_distortion():
        # TODO: Import CV for test-only and match results?
        res = camera_calibration_and_3d_reconstruction.projectPoints()

    def test_projectPoints():
        pass


if __name__ == '__main__':
    unittest.main()
