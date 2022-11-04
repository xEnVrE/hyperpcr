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


std::tuple<bool, MatrixXd, Matrix<unsigned char, Dynamic, Dynamic>> PointCloud::points(const bool& blocking)
{
    auto false_tuple = std::make_tuple(false, Eigen::MatrixXd(), Eigen::Matrix<unsigned char, Dynamic, Dynamic>());

    bool valid = port_.freeze(blocking);
    if (!valid)
        return false_tuple;

    MatrixXd points = port_.matrix_as_double().leftCols(3);
    points.transposeInPlace();

    Matrix<unsigned char, -1, -1> colors(3, points.cols());
    for (std::size_t i = 0; i < points.cols(); i++)
    {
        colors(0, i) = color_[2];
        colors(1, i) = color_[1];
        colors(2, i) = color_[0];
    }

    return std::make_tuple(true, points, colors);
}
