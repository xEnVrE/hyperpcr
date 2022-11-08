import copy
import numpy
import pyquaternion


class PoseFilter():

    def __init__(self):

        self.pose = None


    def step(self, pose):

        if self.pose is None:
            self.pose = pose
        else:
            self.pose = self.filter(pose)

        return self.pose


    def filter(self, pose):
        """Thanks @fceola for the idea."""

        axis_0 = pose[0:3, 0]
        axis_1 = pose[0:3, 1]
        axis_2 = pose[0:3, 2]

        rotations = []
        for i in range(2):

            if i == 0:
                x = axis_0
            else:
                x = -axis_0

            for j in range(2):
                if j == 0:
                    y = axis_1
                else:
                    y = -axis_1

                rotation = numpy.zeros((3, 3))
                rotation[0:3, 0] = x
                rotation[0:3, 1] = y
                rotation[0:3, 2] = numpy.cross(x, y)
                rotations.append(rotation)

        best_rotation = self.find_best_rotation(rotations)
        pose_copy = copy.deepcopy(pose)
        pose_copy[0:3, 0:3] = best_rotation

        return pose_copy


    def find_best_rotation(self, rotations):

        errors = []
        ref_rotation = self.pose[0:3, 0:3]
        for i in range(len(rotations)):
            q = pyquaternion.Quaternion(matrix = ref_rotation @ rotations[i].T)
            error = numpy.linalg.norm(q.axis * q.angle)
            errors.append(error)

        return rotations[numpy.argmin(errors)]
