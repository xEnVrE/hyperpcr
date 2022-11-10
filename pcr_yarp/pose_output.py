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

        output_vector = numpy.zeros(7)
        if pose is not None:
            q = pyquaternion.Quaternion(matrix = pose[0:3, 0:3])

            output_vector[0:3] = pose[0:3, 3]
            output_vector[3:6] = q.axis
            output_vector[6] = q.angle

        output_vector_yarp = yarp.Vector()
        output_vector_yarp.resize(7 + points.shape[0] * points.shape[1])
        for i in range(7):
            output_vector_yarp[i] = output_vector[i]

        # The receiver will treat these as the 8 vertices of the object oriented bounding box
        points_offset = 7
        for i in range(8):
            base_offset = points_offset + i * 3
            output_vector_yarp[base_offset + 0] = points[i, 0]
            output_vector_yarp[base_offset + 1] = points[i, 1]
            output_vector_yarp[base_offset + 2] = points[i, 2]

        # TODO: add stamp propagation

        self.pose_out.write(output_vector_yarp)
