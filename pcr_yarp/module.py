import argparse
import copy
import inspect
import open3d as o3d
import os
import numpy
import pcr
import pyquaternion
import torch
import yarp
from dbscan import DBSCAN
from pcr.model import PCRNetwork as Model
from pcr.utils import Normalize, Denormalize
from pcr.default_config import Config
from pcr_yarp.config import Config as IMConfig
from pcr_yarp.pose_filter import PoseFilter


class InferenceModule(yarp.RFModule):

    def __init__(self, config):

        self.config = config

        # Set requested GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config.Module.gpu_id)

        # Initialize YARP network
        yarp.Network.init()

        # Initialize RF module
        yarp.RFModule.__init__(self)

        # Initialize inference
        module_path = '/'.join(inspect.getsourcefile(pcr).split('/')[:-2])
        ckpt_path = os.path.join(module_path, 'checkpoints', 'grasping.ckpt')
        self.model = Model(config = Config.Model)
        self.model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        self.model.cuda()
        self.model.eval()

        # Initialize pose filter
        self.pose_filter = PoseFilter()

        # Initialize YARP ports
        self.depth_in = yarp.BufferedPortImageFloat()
        self.depth_in.open('/hyperpcr/depth:i')

        self.mask_in = yarp.BufferedPortImageMono()
        self.mask_in.open('/hyperpcr/mask:i')

        self.cloud_out = yarp.Port()
        self.cloud_out.open('/hyperpcr/cloud:o')

        # Input buffers initialization
        self.depth_buffer = bytearray(numpy.zeros((self.config.Camera.height, self.config.Camera.width, 1), dtype = numpy.float32))
        self.depth_image = yarp.ImageFloat()
        self.depth_image.resize(self.config.Camera.width, self.config.Camera.height)
        self.depth_image.setExternal(self.depth_buffer, self.config.Camera.width, self.config.Camera.height)

        self.mask_buffer = bytearray(numpy.zeros((self.config.Camera.height, self.config.Camera.width, 1), dtype = numpy.uint8))
        self.mask_image = yarp.ImageMono()
        self.mask_image.resize(self.config.Camera.width, self.config.Camera.height)
        self.mask_image.setExternal(self.mask_buffer, self.config.Camera.width, self.config.Camera.height)

        self.store_image_selector()


    def store_image_selector(self):

        fx = float(self.config.Camera.fx)
        fy = float(self.config.Camera.fy)
        cx = float(self.config.Camera.cx)
        cy = float(self.config.Camera.cy)
        self.selector_u = numpy.zeros((self.config.Camera.height, self.config.Camera.width), dtype = numpy.float32)
        self.selector_v = numpy.zeros((self.config.Camera.height, self.config.Camera.width), dtype = numpy.float32)
        for v in range(self.config.Camera.height):
            for u in range(self.config.Camera.width):
                self.selector_u[v, u] = (u - cx) / fx
                self.selector_v[v, u] = (v - cy) / fy


    def close(self):

        self.depth_in.close()
        self.mask_in.close()

        return True


    def getPeriod(self):

        return self.config.Module.period


    def get_obb_pose(self, points):

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        bbox = cloud.get_oriented_bounding_box()
        pose = pyquaternion.Quaternion(matrix = bbox.R).transformation_matrix
        pose[0:3, 3] = cloud.get_center()

        return pose


    def updateModule(self):

        depth = self.depth_in.read(False)
        mask = self.mask_in.read(False)

        if (depth is not None) and (mask is not None):

            starter = torch.cuda.Event(enable_timing = True)
            ender = torch.cuda.Event(enable_timing = True)
            starter.record()

            self.depth_image.copy(depth)
            depth_frame = numpy.frombuffer(self.depth_buffer, dtype=numpy.float32).reshape(self.config.Camera.height, self.config.Camera.width)

            self.mask_image.copy(mask)
            mask_frame = numpy.frombuffer(self.mask_buffer, dtype=numpy.uint8).reshape(self.config.Camera.height, self.config.Camera.width)

            mask_selector = mask_frame != 0
            depth_valid_up_selector = depth_frame < self.config.Depth.upper_bound
            depth_valid_down_selector = depth_frame > self.config.Depth.lower_bound
            valid_selector = mask_selector & depth_valid_up_selector & depth_valid_down_selector
            z = depth_frame[valid_selector]
            x_z = self.selector_u[valid_selector]
            y_z = self.selector_v[valid_selector]

            if len(z) > 0:
                cloud = numpy.zeros((z.shape[0], 3), dtype = numpy.float32)
                cloud[:, 0] = x_z * z
                cloud[:, 1] = y_z * z
                cloud[:, 2] = z

                if self.config.DBSCAN.enable:
                    cloud = self.dbscan_filter(cloud)

                partial, ctx = Normalize(Config.Processing)(cloud)
                partial = torch.tensor(partial, dtype=torch.float32).cuda().unsqueeze(0)
                complete, probabilities = self.model(partial)
                complete = complete.squeeze(0).cpu().numpy()
                complete = Denormalize(Config.Processing)(complete, ctx)

                pose = self.get_obb_pose(complete)
                if self.config.PoseFiltering.enable:
                    pose = self.pose_filter.step(pose)

                self.send_output(complete, pose)

            ender.record()
            torch.cuda.synchronize()
            elapsed = starter.elapsed_time(ender) / 1000.0
            print(1.0 / elapsed)

        return True


    def dbscan_filter(self, cloud):

        labels, _ = DBSCAN(cloud.astype(dtype = numpy.float64), eps = self.config.DBSCAN.eps, min_samples = self.config.DBSCAN.min_samples)
        labels_count = [list(labels).count(i) for i in range(0, labels.max() + 1)]
        if len(labels_count) > 0:
            label_max = numpy.argmax(labels_count)
            if labels_count[label_max] > 0:
                cloud = cloud[labels == label_max]

        return cloud


    def send_output(self, cloud, pose):

        # convert pose in position, axis, angle
        q = pyquaternion.Quaternion(matrix = pose[0:3, 0:3])
        axis_angle = numpy.zeros(4)
        axis_angle[:3] = q.axis
        axis_angle[3] = q.angle

        # we allocate two additional columns to send also the pose of the oriented bounding box
        total_size = cloud.shape[0] + 2
        self.yarp_cloud_source = numpy.zeros((total_size, 4), dtype = numpy.float32)
        self.yarp_cloud_source[:cloud.shape[0], 0:3] = cloud
        self.yarp_cloud_source[cloud.shape[0], 0:3] = pose[0:3, 3]
        self.yarp_cloud_source[cloud.shape[0] + 1, :] = axis_angle

        yarp_cloud = yarp.ImageFloat()
        yarp_cloud.resize(self.yarp_cloud_source.shape[1], self.yarp_cloud_source.shape[0])
        yarp_cloud.setExternal(self.yarp_cloud_source.data, self.yarp_cloud_source.shape[1], self.yarp_cloud_source.shape[0])
        self.cloud_out.write(yarp_cloud)


def main():
    config = IMConfig()
    module = InferenceModule(config)
    module.runModule()


if __name__ == '__main__':
    main()
