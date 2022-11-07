/*
 * Copyright (C) 2022 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * GPL-2+ license. See the accompanying LICENSE file for details.
 */

#include <Viewer.h>
#include <PointCloud.h>

#include <RobotsIO/Camera/Camera.h>
#include <RobotsIO/Camera/CameraParameters.h>
#include <RobotsIO/Camera/RealsenseCameraYarp.h>
#include <RobotsIO/Camera/YarpCamera.h>

#include <RobotsViz/VtkPointCloud.h>
#include <RobotsViz/PointCloudCamera.h>
#include <RobotsViz/PointCloudSource.h>

using namespace RobotsIO::Camera;
using namespace RobotsViz;
using namespace yarp::os;


Viewer::Viewer(const ResourceFinder& resource_finder)
{
    const std::string port_prefix = "hyperpcr-viewer";
    const double fps = resource_finder.check("fps", Value(30.0)).asFloat64();

    const Bottle& camera_bottle = resource_finder.findGroup("CAMERA");
    if (camera_bottle.isNull())
        throw(std::runtime_error(log_name_ + "::ctor. Malformed configuration file: cannot find CAMERA section."));

    const std::string camera_source = camera_bottle.check("source", Value("YARP")).asString();

    auto camera_width = camera_bottle.find("width");
    auto camera_height = camera_bottle.find("height");
    auto camera_fx = camera_bottle.find("fx");
    auto camera_fy = camera_bottle.find("fy");
    auto camera_cx = camera_bottle.find("cx");
    auto camera_cy = camera_bottle.find("cy");

    bool valid_camera_values = !(camera_width.isNull()) && camera_width.isInt32();
    valid_camera_values &= !(camera_height.isNull()) && camera_height.isInt32();
    valid_camera_values &= !(camera_fx.isNull()) && camera_fx.isFloat64();
    valid_camera_values &= !(camera_fy.isNull()) && camera_fy.isFloat64();
    valid_camera_values &= !(camera_cx.isNull()) && camera_cx.isFloat64();
    valid_camera_values &= !(camera_cy.isNull()) && camera_cy.isFloat64();

    const Bottle& reconstruction_bottle = resource_finder.findGroup("RECONSTRUCTION");
    if (reconstruction_bottle.isNull())
        throw(std::runtime_error(log_name_ + "::ctor. Malformed configuration file: cannot find RECONSTRUCTION section."));
    auto color = load_vector_uchar(reconstruction_bottle, "color", 3);
    auto reference_frame_length = reconstruction_bottle.check("reference_frame_length", Value(0.1)).asFloat64();

    std::unique_ptr<Camera> camera;
    if (camera_source == "YARP")
    {
        if (!valid_camera_values)
        throw(std::runtime_error(log_name_ + "::ctor. Camera parameters from configuration are invalid."));

        camera = std::make_unique<YarpCamera>
        (
            camera_width.asInt32(),
            camera_height.asInt32(),
            camera_fx.asFloat64(),
            camera_cx.asFloat64(),
            camera_fy.asFloat64(),
            camera_cy.asFloat64(),
            port_prefix, /* enable camera pose input */false
        );
    }
    else if (camera_source == "RealsenseCamera")
        camera = std::make_unique<RealsenseCameraYarp>(port_prefix);
    else
        throw(std::runtime_error(log_name_ + "::ctor. Camera " + camera_source + " is not supported."));

    /* Initialize camera based point cloud. */
    const double far_plane = camera_bottle.check("far_plane", Value(10.0)).asFloat64();
    const double subsampling_radius = camera_bottle.check("subsampling_radius", Value(-1)).asFloat64();
    auto pc = std::make_unique<PointCloudCamera>(std::move(camera), far_plane, subsampling_radius);

    /* Initialize reconstructed point cloud source. */
    auto reconstructed_pc = std::make_unique<PointCloud>("/" + port_prefix + "/reconstructed_cloud:i", color);

    /* Initialize the VTK container and add the clouds */
    auto vtk_pc = std::make_unique<VtkPointCloud>(std::move(pc));
    auto vtk_reconstructed_pc = std::make_unique<VtkPointCloud>(std::move(reconstructed_pc));

    /* Configure the reference frame attached to the reconstructed cloud. */
    vtk_reconstructed_pc->get_reference_frame().set_visibility(true);
    vtk_reconstructed_pc->get_reference_frame().set_length(reference_frame_length);

    vtk_container_ = std::make_unique<VtkContainer>(1.0 / fps, 600, 600, false);
    vtk_container_->add_content("point_cloud", std::move(vtk_pc));
    vtk_container_->add_content("reconstruction", std::move(vtk_reconstructed_pc));
}


void Viewer::run()
{
    vtk_container_->run();
}


std::vector<unsigned char> Viewer::load_vector_uchar(const Bottle& resource, const std::string& key, const std::size_t size)
{
    if (resource.find(key).isNull())
        throw(std::runtime_error(log_name_ + "::load_vector_uchar. Cannot find key " + key + "."));

    Bottle* b = resource.find(key).asList();
    if (b == nullptr)
        throw(std::runtime_error(log_name_ + "::load_vector_uchar. Cannot get vector having key " + key + " as a list."));

    if (b->size() != size)
        throw(std::runtime_error(log_name_ + "::load_vector_uchar. Vector having key " + key + " has size "  + std::to_string(b->size()) + " (expected is " + std::to_string(size) + ")."));

    std::vector<unsigned char> vector(size);
    for (std::size_t i = 0; i < b->size(); i++)
    {
        Value item_v = b->get(i);
        if (item_v.isNull())
            throw(std::runtime_error(log_name_ + "::load_vector_uchar." + std::to_string(i) + "-th element of of vector having key " + key + " is null."));

        if (!item_v.isInt32())
            throw(std::runtime_error(log_name_ + "::load_vector_uchar." + std::to_string(i) + "-th element of of vector having key " + key + " is not a integer."));

        vector[i] = item_v.asInt32();
    }

    return vector;
}
