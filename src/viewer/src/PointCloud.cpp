/*
 * Copyright (C) 2022 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * GPL-2+ license. See the accompanying LICENSE file for details.
 */

#include <PointCloud.h>

#include <iostream>
#include <fstream>

using namespace Eigen;


PointCloud::PointCloud(const std::string& port_name, const std::vector<unsigned char>& color) :
    port_(port_name),
    color_(color)
{}



bool PointCloud::freeze(const bool& blocking)
{

    if (!port_.freeze(blocking))
        return false;

    MatrixXd data = port_.matrix_as_double();
    points_ = data.leftCols(3).topRows(data.rows() - 2);
    points_.transposeInPlace();

    colors_ = Matrix<unsigned char, Dynamic, Dynamic>(3, points_.cols());
    for (std::size_t i = 0; i < points_.cols(); i++)
    {
        colors_(0, i) = color_[2];
        colors_(1, i) = color_[1];
        colors_(2, i) = color_[0];
    }

    return true;
}


MatrixXd PointCloud::points()
{
    return points_;
}


Matrix<unsigned char, Dynamic, Dynamic> PointCloud::colors()
{
    return colors_;
}
