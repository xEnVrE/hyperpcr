import numpy
import pyquaternion
import yarp


class CloudOutput():

    def __init__(self, prefix, config):

        self.config = config

        self.cloud_out = yarp.Port()
        self.cloud_out.open('/' + prefix + '/cloud:o')


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


    def close(self):

        self.cloud_out.close()
