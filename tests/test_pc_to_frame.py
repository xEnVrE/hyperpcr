import numpy
import open3d as o3d

def add_point_cloud(name, points, color, size, scene):

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.paint_uniform_color(color)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultUnlit'
    material.point_size = size

    scene.add_geometry(name, cloud, material)


def main():
    cloud = numpy.loadtxt('./reconstruction.txt')
    centroid = numpy.zeros((1, 3))
    centroid[0, :] = numpy.mean(cloud, axis = 0)

    try:
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        window = app.create_window("Open3d", 1024, 768)
        widget3d = o3d.visualization.gui.SceneWidget()
        widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
        widget3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
        window.add_child(widget3d)

        add_point_cloud('object', cloud, [33 / 255, 150 / 255, 243 / 255], 3, widget3d.scene)
        add_point_cloud('centroid', centroid, [1, 0, 0], 10,  widget3d.scene)

        widget3d.setup_camera(60, widget3d.scene.bounding_box, [0, 0, 0])

        app.run()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
