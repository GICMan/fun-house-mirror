import numpy as np
import cv2


class TPS:
    def __init__(self, frame_size, mesh_size):
        self.frame_width, self.frame_height = frame_size
        self.mesh_width, self.mesh_height = mesh_size

        # create a mesh to warp
        xs = np.linspace(0, self.frame_width-1, self.mesh_width)
        ys = np.linspace(0, self.frame_height-1, self.mesh_height)
        self.mesh_x, self.mesh_y = np.meshgrid(xs, ys)

        # create a coordinate matrix for multiplication later
        self.mesh_coord_mat = np.vstack([
            np.ones((1, self.mesh_width * self.mesh_height)),
            self.mesh_x.ravel(),
            self.mesh_y.ravel(),
        ]).T

    def update_src_points(self, src_points):
        self.src_points = src_points
        self.num_points = self.src_points.shape[0]

        # the TPS method uses a kernal mat based on the TPS function,
        # along with an affine part

        diff = src_points[:, None, :] - src_points[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist = np.where(dist == 0, 1e-8, dist)
        kernal_mat = dist**2 * np.log(dist)

        affine_mat = np.hstack([
            np.ones((self.num_points, 1)),
            self.src_points
        ])

        # combine them to create the constrained system matrix
        tps_mat = np.vstack([
            np.hstack([kernal_mat, affine_mat]),
            np.hstack([affine_mat.T, np.zeros((3, 3))]),
        ])

        # compute the inverse to solve for parameters, add regularization term
        # to ensure stability
        reg = 1e-6 * np.eye(tps_mat.shape[0])
        self.tps_mat = tps_mat + reg

        # compute the base tps function values for each mesh points
        dx = self.mesh_x[..., None] - src_points[:, 0]
        dy = self.mesh_y[..., None] - src_points[:, 1]
        dist = np.sqrt(dx**2 + dy**2)
        dist = np.where(dist == 0, 1e-8, dist)
        mesh_tps = dist**2 * np.log(dist)

        self.flat_mesh_tps = mesh_tps.reshape((-1, self.num_points))

    def compute_map(self, dst_points):
        params_x = np.linalg.solve(self.tps_mat, np.concatenate(
            [dst_points[:, 0], np.zeros(3)]))
        params_y = np.linalg.solve(self.tps_mat, np.concatenate(
            [dst_points[:, 1], np.zeros(3)]))

        tps_params_x = params_x[:self.num_points]
        affine_params_x = params_x[self.num_points:]

        tps_params_y = params_y[:self.num_points]
        affine_params_y = params_y[self.num_points:]

        mesh_map_x = self.flat_mesh_tps.dot(
            tps_params_x) + self.mesh_coord_mat.dot(affine_params_x)
        mesh_map_y = self.flat_mesh_tps.dot(tps_params_y) + \
            self.mesh_coord_mat.dot(affine_params_y)

        full_map_x = cv2.resize(
            mesh_map_x.reshape(
                (self.mesh_width, self.mesh_height)).astype(np.float32),
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_LINEAR
        )
        full_map_y = cv2.resize(
            mesh_map_y.reshape((self.mesh_width, self.mesh_height)
                               ).astype(np.float32),
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_LINEAR
        )

        return full_map_x, full_map_y
