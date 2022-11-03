import time
import numpy
import open3d as o3d
from sklearn.decomposition import PCA


def add_point_cloud(name, points, color, size, scene):

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.paint_uniform_color(color)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultUnlit'
    material.point_size = size

    scene.add_geometry(name, cloud, material)


def add_axes(name, pose, scene):

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.05)
    frame.transform(pose)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultUnlit'

    scene.add_geometry(name, frame, material)


def get_pose(points):

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    center = cloud.get_center()
    bbox = cloud.get_oriented_bounding_box()
    pose = numpy.eye(4)
    pose[0:3, 0:3] = bbox.R
    pose[0:3, 3] = center

    return pose


def main():
    cloud_in = numpy.loadtxt('./reconstruction.txt')
    cloud_out = numpy.loadtxt('./reconstruction.txt')

    pose = get_pose(cloud_out)

    centroid = numpy.zeros((1, 3))
    centroid[0, :] = pose[0:3, 3]

    try:
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        window = app.create_window("Open3d", 1024, 768)
        widget3d = o3d.visualization.gui.SceneWidget()
        widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
        widget3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
        window.add_child(widget3d)

        add_point_cloud('partial', cloud_in, [100 / 255, 10 / 255, 10 / 255], 3, widget3d.scene)
        add_point_cloud('object', cloud_out, [33 / 255, 150 / 255, 243 / 255], 3, widget3d.scene)
        add_point_cloud('centroid', centroid, [1, 0, 0], 10, widget3d.scene)
        widget3d.setup_camera(60, widget3d.scene.bounding_box, [0, 0, 0])
        add_axes('camera_axes', numpy.eye(4), widget3d.scene)
        add_axes('object_axes', pose, widget3d.scene)

        app.run()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
