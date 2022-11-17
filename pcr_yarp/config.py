class Config:

    class Camera:
        width = int(640)
        height = int(480)
        fx = float(618.0714111328125)
        fy = float(617.783447265625)
        cx = float(305.902252197265625)
        cy = float(246.352935791015625)

    class DBSCAN:
        enable = bool(True)
        eps = float(0.01)
        min_samples = int(100)
        use_cuml = bool(True)

    class Depth:
        lower_bound = float(0.2)
        upper_bound = float(1.0)

    class Input:
        joint_input_mode = bool(False)

    class Module:
        gpu_id = int(0)
        period = float(0.01)

    class Open3D:
        oriented_bounding_box_robust = bool(True)

    class PoseFiltering:
        enable = bool(True)

    def __str__(self):

        c = self.Camera
        d = self.DBSCAN
        de = self.Depth
        i = self.Input
        m = self.Module
        o = self.Open3D
        p = self.PoseFiltering

        return 'Camera parameters are:\r\n' + \
               '    w = ' + str(c.width) + ', h = ' + str(c.height) + ', fx = ' + str(c.fx) + ', fy = ' + str(c.fy) + ', cx = ' + str(c.cx) + ', cy = ' + str(c.cy) + '\r\n\r\n' + \
               'DBSCAN parameters are:\r\n' + \
               '    enable = ' + str(d.enable) + ', eps = ' + str(d.eps) + ', min_samples = ' + str(d.min_samples) + ', use_cuml = ' + str(d.use_cuml) + '\r\n\r\n' + \
               'Depth parameters are:\r\n' + \
               '    lower_bound = ' + str(de.lower_bound) + ', upper_bound = ' + str(de.upper_bound) + '\r\n\r\n' + \
               'Input paramters are:\r\n' + \
               '    joint_input_mode = ' + str(i.joint_input_mode) + '\r\n\r\n' + \
               'Module parameters are:\r\n' + \
               '    gpu_id = ' + str(m.gpu_id) + ', period = ' + str(m.period) + '\r\n\r\n' + \
               'Open3D parameters are:\r\n' + \
               '    oriented_bounding_box_robust = ' + str(o.oriented_bounding_box_robust) + '\r\n\r\n' + \
               'Pose filtering parameters are:\r\n' + \
               '    enable = ' + str(p.enable) + '\r\n'
