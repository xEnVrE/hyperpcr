/*
 * Copyright (C) 2022 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * GPL-2+ license. See the accompanying LICENSE file for details.
 */

#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <RobotsViz/PointCloudSource.h>

#include <RobotsIO/Utils/FloatMatrixYarpPort.h>

#include <memory>

namespace RobotsViz {
    class PointCloudCamera;
}


class PointCloud : public RobotsViz::PointCloudSource
{
public:
    PointCloud(const std::string& port_name, const std::vector<unsigned char>& color);

    std::tuple<bool, Eigen::MatrixXd, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>> points(const bool& blocking) override;

private:
    RobotsIO::Utils::FloatMatrixYarpPort port_;

    const std::vector<unsigned char> color_;
};

#endif /* POINTCLOUD_H */
