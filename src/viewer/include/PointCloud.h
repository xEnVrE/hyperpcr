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
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PointCloud(const std::string& port_name, const std::vector<unsigned char>& color);

    bool freeze(const bool& blocking);

    Eigen::MatrixXd points()  override;

    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> colors()  override;

    std::tuple<bool, Eigen::Transform<double, 3, Eigen::Affine>> pose() override;

private:
    RobotsIO::Utils::FloatMatrixYarpPort port_;

    const std::vector<unsigned char> color_;

    Eigen::MatrixXd points_;

    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> colors_;

    Eigen::Transform<double, 3, Eigen::Affine> pose_;
};

#endif /* POINTCLOUD_H */
