/*
 * Copyright (C) 2022 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * GPL-2+ license. See the accompanying LICENSE file for details.
 */

#ifndef VIEWER_H
#define VIEWER_H

#include <RobotsViz/VtkContainer.h>

#include <string>
#include <vector>

#include <yarp/os/Bottle.h>
#include <yarp/os/ResourceFinder.h>


class Viewer
{
public:
    Viewer(const yarp::os::ResourceFinder& resource_finder);

    void run();

private:
    std::vector<unsigned char> load_vector_uchar(const yarp::os::Bottle& resource, const std::string& key, const std::size_t size);

    std::unique_ptr<RobotsViz::VtkContainer> vtk_container_;

    const std::string log_name_ = "hyperpcr-viewer";
};

#endif /* VIEWER_H */
