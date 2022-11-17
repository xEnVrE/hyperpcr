import argparse
import copy
import cuml
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
from pcr_yarp.cloud_output import CloudOutput
from pcr_yarp.image_input import ImageInput
from pcr_yarp.pose_filter import PoseFilter
from pcr_yarp.pose_output import PoseOutput


class InferenceModule(yarp.RFModule):

    def __init__(self, config):

        self.config = config

        # Set requested GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config.Module.gpu_id)
        cuml.set_global_output_type('numpy')

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

        # Initialize image input
        self.image_input = ImageInput('hyperpcr', config)

        # Initialize point cloud output
        self.cloud_output = CloudOutput('hyperpcr', config)

        # Initialize pose output
        self.pose_output = PoseOutput('hyperpcr', config)

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

        self.image_input.close()
        self.cloud_output.close()
        self.pose_output.close()

        return True


    def getPeriod(self):

        return self.config.Module.period


    def get_obb_data(self, points):

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        # This is required as the underlying Qhull might fail when points are distributed weirdly in space
        try:
            bbox = cloud.get_oriented_bounding_box()
        except RuntimeError:
            print('Warning: qhull failed. Cannot extract the pose from the point cloud.')
            return False, None, None

        pose = pyquaternion.Quaternion(matrix = bbox.R).transformation_matrix
        pose[0:3, 3] = cloud.get_center()

        points = numpy.asarray(bbox.get_box_points())

        return True, pose, points


    def updateModule(self):

        valid_images, depth, mask = self.image_input.get_images()

        if valid_images:
            try:
                starter = torch.cuda.Event(enable_timing = True)
                ender = torch.cuda.Event(enable_timing = True)
                starter.record()

                valid_cloud, cloud = self.get_point_cloud(depth, mask)

                if valid_cloud:
                    if self.config.DBSCAN.enable:
                        cloud = self.dbscan_filter(cloud)

                    complete = self.complete_cloud(cloud)

                    valid_obb, pose, points = self.get_obb_data(complete)
                    if valid_obb:
                        if self.config.PoseFiltering.enable:
                            pose = self.pose_filter.step(pose)

                        self.pose_output.send_output(pose, points)
                        self.cloud_output.send_output(complete, pose)
                    else:
                        # Send a non valid pose to mark that the input was received but the output is not available
                        self.pose_output.send_output(None, None)
                else:
                    # Send a non valid pose to mark that the input was received but the output is not available
                    self.pose_output.send_output(None, None)

                ender.record()
                torch.cuda.synchronize()
                elapsed = starter.elapsed_time(ender) / 1000.0
                print(1.0 / elapsed)

            except Exception as e:
                print(e)
                print("An exception has occured, sending a invalid pose.")

                # Send a non valid pose to mark that the input was received but the output is not available
                self.pose_output.send_output(None, None)

        return True


    def dbscan_filter(self, cloud):

        if self.config.DBSCAN.use_cuml:
            dbscan = cuml.DBSCAN(eps = self.config.DBSCAN.eps, min_samples = self.config.DBSCAN.min_samples)
            dbscan.fit(cloud)
            labels = dbscan.labels_
        else:
            labels, _ = DBSCAN(cloud.astype(dtype = numpy.float64), eps = self.config.DBSCAN.eps, min_samples = self.config.DBSCAN.min_samples)

        labels_count = [list(labels).count(i) for i in range(0, labels.max() + 1)]
        if len(labels_count) > 0:
            label_max = numpy.argmax(labels_count)
            if labels_count[label_max] > 0:
                cloud = cloud[labels == label_max]

        return cloud


    def get_point_cloud(self, depth, mask):

        mask_selector = mask != 0
        depth_valid_up_selector = depth < self.config.Depth.upper_bound
        depth_valid_down_selector = depth > self.config.Depth.lower_bound
        valid_selector = mask_selector & depth_valid_up_selector & depth_valid_down_selector
        z = depth[valid_selector]
        x_z = self.selector_u[valid_selector]
        y_z = self.selector_v[valid_selector]

        if len(z) > 0:
            cloud = numpy.zeros((z.shape[0], 3), dtype = numpy.float32)
            cloud[:, 0] = x_z * z
            cloud[:, 1] = y_z * z
            cloud[:, 2] = z

            return True, cloud
        else:
            return False, None


    def complete_cloud(self, cloud):

        partial, ctx = Normalize(Config.Processing)(cloud)
        partial = torch.tensor(partial, dtype=torch.float32).cuda().unsqueeze(0)
        complete, probabilities = self.model(partial)
        complete = complete.squeeze(0).cpu().numpy()
        complete = Denormalize(Config.Processing)(complete, ctx)

        return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--override-config-file', dest = 'override_config_file', action = 'store_true')
    parser.add_argument('--use-joint-input-mode', dest = 'use_joint_input_mode', action = 'store_true')
    parser.set_defaults(override_config_file = False)
    parser.set_defaults(use_joint_input_mode= False)
    args = parser.parse_args()

    config = IMConfig()

    if args.override_config_file:
        if args.use_joint_input_mode:
            config.Input.joint_input_mode = True

    module = InferenceModule(config)
    module.runModule()


if __name__ == '__main__':
    main()
