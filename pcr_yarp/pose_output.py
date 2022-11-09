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


    def send_output(self, pose):

        output_vector = numpy.zeros(7)
        if pose is not None:
            q = pyquaternion.Quaternion(matrix = pose[0:3, 0:3])

            output_vector[0:3] = pose[0:3, 3]
            output_vector[3:6] = q.axis
            output_vector[6] = q.angle

        output_vector_yarp = yarp.Vector()
        output_vector_yarp.resize(7)
        for i in range(7):
            output_vector_yarp[i] = output_vector[i]

        # TODO: add stamp propagation

        self.pose_out.write(output_vector_yarp)
