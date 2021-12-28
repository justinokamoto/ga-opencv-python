import numpy as np
import cupy as cp


"""
This module implements routines defined within:
https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
"""

# TODO: swap intrinsics/extrinsics with vectors
def projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs):
    """
    TODO: Argument documentation
    
    Python: cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs[, imagePoints[, jacobian[, aspectRatio]]]) → imagePoints, jacobian

    Projects 3D points to an image plane.

    https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#projectpoints
    """
    # TODO: Instead of host_ convention, switch to _gpu convention
    # TODO: Ensure determinism?
    # def project_points(self,
    #                    host_pointcloud: np.ndarray,
    #                    host_intrinsics: Intrinsics,
    #                    host_extrinsics: Extrinsics,
    #                    rgb_shape: Tuple[int, int, int]):

    # TODO: Is this efficient in terms of GPU mem pages?
    pointcloud = cp.asarray(host_pointcloud) # Move from host to device
    distortion_coeffs = cp.asarray(host_intrinsics.distortion) # Move from host to device
    camera_matrix = cp.asarray(host_intrinsics.camera_matrix) # Move from host to device
    translation = cp.asarray(host_extrinsics.translation) # Move from host to device
    rotation = cp.asarray(host_extrinsics.rotation) # Move from host to device
    # Allocate output structure
    # TODO: Explain this structure
    aligned_depth = cp.full((*rgb_shape[:2], 3), cp.nan, dtype=cp.float32)
    # Copy PC array which will be modified in place to convert from
    # x,y,z coords to u,v,_
    pixel_coords = pointcloud.copy()

    # ----- compute pixel coords -----

    # Apply camera extrinsics [R|t] to pointcloud points
    # TODO: Can this be broadcasted better?
    pixel_coords[:,0] = rotation[0][0] * pointcloud[:,0] + \
                        rotation[0][1] * pointcloud[:,1] + \
                        rotation[0][2] * pointcloud[:,2] + \
                        translation[0]
    pixel_coords[:,1] = rotation[1][0] * pointcloud[:,0] + \
                        rotation[1][1] * pointcloud[:,1] + \
                        rotation[1][2] * pointcloud[:,2] + \
                        translation[1]
    pixel_coords[:,2] = rotation[2][0] * pointcloud[:,0] + \
                        rotation[2][1] * pointcloud[:,1] + \
                        rotation[2][2] * pointcloud[:,2] + \
                        translation[2]
    # Calculate x' and y'
    pixel_coords[:,0] = pixel_coords[:,0]/pixel_coords[:,2]
    pixel_coords[:,1] = pixel_coords[:,1]/pixel_coords[:,2]

    k1, k2, p1, p2, k3 = distortion_coeffs

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
    pixel_coords[:,0] = camera_matrix[0][0] * pixel_coords[:,0] + camera_matrix[0][2]
    pixel_coords[:,1] = camera_matrix[1][1] * pixel_coords[:,1] + camera_matrix[1][2]

    # Round float values to nearest int
    cp.round(pixel_coords, out=pixel_coords)
    # Pixel coords should be integers
    pixel_coords = pixel_coords.astype(cp.int)

    # TODO: Coerce back into OpenCV format instead!


    # ----- coerce pixel_coords into aligned_depth structure ------

    # pixel_x = pixel_coords[:, 0]
    # pixel_y = pixel_coords[:, 1]

    # # Predicate that filters out pixel coords outside the image
    # predicate = ((pixel_y >= 0) & (pixel_y < rgb_shape[1]) & \
    #              (pixel_x >= 0) & (pixel_x < rgb_shape[0]))
    # # Valid pixel coords
    # pixel_x_inrange = pixel_coords[predicate][:, 0]
    # pixel_y_inrange = pixel_coords[predicate][:, 1]
    # # For valid pixel coords, set value to pointcloud point
    # aligned_depth[pixel_x_inrange, pixel_y_inrange] = pointcloud[predicate]
    # host_aligned_depth = cp.asnumpy(aligned_depth) # Move from device to host

    return host_aligned_depth
