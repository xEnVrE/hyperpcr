import numpy
import pyquaternion
import yarp


class PoseOutput():

    def __init__(self, prefix, config):

        self.config = config

        self.pose_out = yarp.Port()
        self.pose_out.open('/' + prefix + '/pose:o')


    def close(self):

        self.pose_out.close()


    def send_output(self, pose, points):

        total_size = 7
        if points is not None:
            total_size += points.shape[0] * points.shape[1]

        output_vector = numpy.zeros(total_size)
        if pose is not None:
            q = pyquaternion.Quaternion(matrix = pose[0:3, 0:3])

            output_vector[0:3] = pose[0:3, 3]
            output_vector[3:6] = q.axis
            output_vector[6] = q.angle

        # The receiver will treat these as the 8 vertices of the object oriented bounding box
        if points is not None:
            for i in range(points.shape[0]):
                base_offset = 7 + i * 3
                output_vector[base_offset + 0] = points[i, 0]
                output_vector[base_offset + 1] = points[i, 1]
                output_vector[base_offset + 2] = points[i, 2]

        output_vector_yarp = yarp.Vector()
        output_vector_yarp.resize(total_size)
        for i in range(total_size):
            output_vector_yarp[i] = output_vector[i]

        # TODO: add stamp propagation

        self.pose_out.write(output_vector_yarp)
