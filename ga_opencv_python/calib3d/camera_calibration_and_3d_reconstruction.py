import numpy as np
import cupy as cp

"""
This module implements routines defined within:
https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
"""
def projectPoints(*args, host_memory=True, **kwargs):
    """
    GPU implementation of https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#projectpoints

    Additional Arguments:
      host_mem (bool): Result array will be in host memory
    """
    res = cp.asnumpy(_projectPoints(*args, **kwargs)) if host_memory else \
        _projectPoints(*args, **kwargs)
    return res


# TODO: API compatibility w/ cv2:
#       * optional aspect ratio parameter
#       * return jacobian
#       * floating point precision causing result differences?
def _projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs):
    """
    Drop-in replacement for https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#projectpoints
    """
    # TODO: distCoeffs validate
    #   must require argument (like cv2)
    #   validate type is float64 or float32
    #   error if rows != 1 and cols != 1
    #   support 12 or 14?
    # Validate number of distortion coefficients
    if distCoeffs is None:
        distCoeffs = np.zeros(8, dtype=np.float64)
    elif len(distCoeffs) not in {4, 5, 8}:
        # TODO: Match cv2 exception
        raise ValueError("Number of distortion coefficients must be 4, 5, or 8 but" \
                         "received %d" % len(distCoeffs))
    elif len(distCoeffs) != 8:
        # Re-instantiate `distCoeffs` parameter to have set length 8
        # (max number of distortion coefficients). Otherwise, this
        # variable length array would require extra length checks.
        allDistCoeffs = np.zeros(8, dtype=np.float64)
        np.copyto(allDistCoeffs[:len(distCoeffs)], distCoeffs)
        distCoeffs = allDistCoeffs
    # Move data from host to device
    object_points_gpu = cp.asarray(objectPoints)
    dist_coeffs_gpu = cp.asarray(distCoeffs)
    camera_matrix_gpu = cp.asarray(cameraMatrix)
    tvec_gpu = cp.asarray(tvec)
    rotation_gpu = cp.asarray(rvec)

    # Copy PC array which will be modified in place to convert from
    # x,y,z coords to u,v,_
    pixel_coords = object_points_gpu.copy()

    # Apply camera extrinsics [R|t] to object_points_gpu points
    # TODO: Can this be broadcasted better?
    pixel_coords[:,0] = rotation_gpu[0][0] * object_points_gpu[:,0] + \
                        rotation_gpu[0][1] * object_points_gpu[:,1] + \
                        rotation_gpu[0][2] * object_points_gpu[:,2] + \
                        tvec_gpu[0]
    pixel_coords[:,1] = rotation_gpu[1][0] * object_points_gpu[:,0] + \
                        rotation_gpu[1][1] * object_points_gpu[:,1] + \
                        rotation_gpu[1][2] * object_points_gpu[:,2] + \
                        tvec_gpu[1]
    pixel_coords[:,2] = rotation_gpu[2][0] * object_points_gpu[:,0] + \
                        rotation_gpu[2][1] * object_points_gpu[:,1] + \
                        rotation_gpu[2][2] * object_points_gpu[:,2] + \
                        tvec_gpu[2]
    # Calculate x' and y'
    pixel_coords[:,0] = pixel_coords[:,0]/pixel_coords[:,2]
    pixel_coords[:,1] = pixel_coords[:,1]/pixel_coords[:,2]

    # TODO: Account distortion for k4, k5, k6
    k1, k2, p1, p2, k3, k4, k5, k6 = dist_coeffs_gpu

    r2 = pixel_coords[:,0] ** 2 + pixel_coords[:,1] ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3

    # Calculate x'' and y''
    pixel_coords[:,0] = pixel_coords[:,0] * (1 + k1 * r2 + k2 * r4 + k3 * r6) + \
                        (2 * p1 * pixel_coords[:,0] * pixel_coords[:,1]) + \
                        p2 * (r2 + 2 * pixel_coords[:,0] ** 2)
    pixel_coords[:,1] = pixel_coords[:,1] * (1 + k1 * r2 + k2 * r4 + k3 * r6) + \
                        (2 * p2 * pixel_coords[:,0] * pixel_coords[:,1]) + \
                        p1 * (r2 + 2 * pixel_coords[:,1] ** 2)

    # Calculate u and v
    pixel_coords[:,0] = camera_matrix_gpu[0][0] * pixel_coords[:,0] + camera_matrix_gpu[0][2]
    pixel_coords[:,1] = camera_matrix_gpu[1][1] * pixel_coords[:,1] + camera_matrix_gpu[1][2]

    # Create array view of just u,v coordinates, wrapping in extra
    # axis to conform to cv2 interface, where returned coords are
    # within 2D array, e.g.:
    # -> [[ [u1, v1], [u2, v2], ... ]]
    pixel_coords = cp.expand_dims(pixel_coords[:,:2], axis=0)
    return pixel_coords, None
