import numpy as np


class RadialCameraModel:
    # simple radial camera model from colmap
    def __init__(self, size, focal_length, principal_point, distortion_coefficients):
        self.size = size
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.distortion_coefficients = distortion_coefficients

    def resize(self, factor):
        self.size = (self.size * factor).astype(int)
        self.focal_length *= factor
        self.principal_point *= factor
        
    def project(self, points3d):
        valid = points3d[:, 2] > 0
        uv = np.zeros_like(points3d[:, :2])
        uv[valid] = points3d[:, :2][valid] / points3d[valid, 2][:, None]

        u2 = uv[:, 0] * uv[:, 0]
        v2 = uv[:, 1] * uv[:, 1]
        r2 = u2 + v2
        radial = 1 + self.distortion_coefficients * r2
        uv *= radial[:, None]
        return uv * self.focal_length + self.principal_point

    def unproject(self, points2d):
        #print(points2d)
        points2d = points2d - self.principal_point
        points2d /= self.focal_length
        r2 = np.sum(points2d ** 2, axis=1)
        factor = 1.0 / (1 + self.distortion_coefficients * r2)
        unprojected = points2d * factor[:, None] 

        #print(unprojected)
        # hom = np.hstack((unprojected, np.ones((unprojected.shape[0], 1))))
        # print(self.project(hom))
        # exit(1)
        return unprojected

    def __str__(self):
        return 'RadialCameraModel(size={}, focal_length={}, principal_point={}, distortion_coefficients={})'.format(self.size, self.focal_length, self.principal_point, self.distortion_coefficients)


def parse_camera_model(text):
    parts = text.split()
    model = parts[0]
    if model == 'SIMPLE_RADIAL':
        size = np.array([int(parts[1]), int(parts[2])])
        focal_length = float(parts[3])
        principal_point = np.array([float(parts[4]), float(parts[5])])
        distortion_coefficients = float(parts[6])
        return RadialCameraModel(size, focal_length, principal_point, distortion_coefficients)
    else:
        raise ValueError('Unknown camera model: {}'.format(model))