/******************************************************************************
* Copyright (c) 2016, Bradley J Chambers (brad.chambers@gmail.com)
*
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following
* conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in
*       the documentation and/or other materials provided
*       with the distribution.
*     * Neither the name of Hobu, Inc. or Flaxen Geo Consulting nor the
*       names of its contributors may be used to endorse or promote
*       products derived from this software without specific prior
*       written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
* AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
* OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
****************************************************************************/

#include <pdal/EigenUtils.hpp>
#include <pdal/PointView.hpp>
#include <pdal/util/Bounds.hpp>
#include <pdal/util/Utils.hpp>

#include <Eigen/Dense>

#include <vector>

namespace pdal
{
  
namespace eigen
{

Eigen::Vector3f computeCentroid(PointView& view, std::vector<PointId> ids)
{
    using namespace Eigen;

    auto n = ids.size();

    double mx, my, mz;
    mx = my = mz = 0.0;
    for (auto const& j : ids)
    {
        mx += view.getFieldAs<double>(Dimension::Id::X, j);
        my += view.getFieldAs<double>(Dimension::Id::Y, j);
        mz += view.getFieldAs<double>(Dimension::Id::Z, j);
    }

    Vector3f centroid;
    centroid << mx/n, my/n, mz/n;

    return centroid;
}

Eigen::Matrix3f computeCovariance(PointView& view, std::vector<PointId> ids)
{
    using namespace Eigen;

    auto n = ids.size();

    Vector3f centroid = computeCentroid(view, ids);

    // demean the neighborhood
    MatrixXf A(3, n);
    size_t k = 0;
    for (auto const& j : ids)
    {
        A(0, k) = view.getFieldAs<double>(Dimension::Id::X, j) - centroid[0];
        A(1, k) = view.getFieldAs<double>(Dimension::Id::Y, j) - centroid[1];
        A(2, k) = view.getFieldAs<double>(Dimension::Id::Z, j) - centroid[2];
        k++;
    }

    return A * A.transpose();
}

uint8_t computeRank(PointView& view, std::vector<PointId> ids, double threshold)
{
    using namespace Eigen;

    Matrix3f B = computeCovariance(view, ids);

    JacobiSVD<Matrix3f> svd(B);
    svd.setThreshold(threshold);

    return static_cast<uint8_t>(svd.rank());
}

Eigen::MatrixXd computeSpline(Eigen::MatrixXd x, Eigen::MatrixXd y,
                              Eigen::MatrixXd z, Eigen::MatrixXd xx,
                              Eigen::MatrixXd yy)
{
    using namespace Eigen;

    int num_rows = xx.rows();
    int num_cols = xx.cols();

    MatrixXd S = MatrixXd::Zero(num_rows, num_cols);

    for (auto col = 0; col < num_cols; ++col)
    {
        for (auto row = 0; row < num_rows; ++row)
        {
            // Further optimizations are achieved by estimating only the
            // interpolated surface within a local neighbourhood (e.g. a 7 x 7
            // neighbourhood is used in our case) of the cell being filtered.
            int radius = 3;

            int c = std::floor(col/2);
            int r = std::floor(row/2);

            int cs = Utils::clamp(c-radius, 0, static_cast<int>(z.cols()-1));
            int ce = Utils::clamp(c+radius, 0, static_cast<int>(z.cols()-1));
            int col_size = ce - cs + 1;
            int rs = Utils::clamp(r-radius, 0, static_cast<int>(z.rows()-1));
            int re = Utils::clamp(r+radius, 0, static_cast<int>(z.rows()-1));
            int row_size = re - rs + 1;

            MatrixXd Xn = x.block(rs, cs, row_size, col_size);
            MatrixXd Yn = y.block(rs, cs, row_size, col_size);
            MatrixXd Hn = z.block(rs, cs, row_size, col_size);

            int nsize = row_size * col_size;
            VectorXd T = VectorXd::Zero(nsize);
            MatrixXd P = MatrixXd::Zero(nsize, 3);
            MatrixXd K = MatrixXd::Zero(nsize, nsize);

            for (auto id = 0; id < nsize; ++id)
            {
                double xj = Xn(id);
                double yj = Yn(id);
                double zj = Hn(id);
                if (std::isnan(xj) || std::isnan(yj) || std::isnan(zj))
                    continue;
                T(id) = zj;
                P.row(id) << 1, xj, yj;
                for (auto id2 = 0; id2 < nsize; ++id2)
                {
                    if (id == id2)
                        continue;
                    double xk = Xn(id2);
                    double yk = Yn(id2);
                    double zk = Hn(id2);
                    if (std::isnan(xk) || std::isnan(yk) || std::isnan(zk))
                        continue;
                    double rsqr = (xj - xk) * (xj - xk) + (yj - yk) * (yj - yk);
                    if (rsqr == 0.0)
                        continue;
                    K(id, id2) = rsqr * std::log10(std::sqrt(rsqr));
                }
            }

            MatrixXd A = MatrixXd::Zero(nsize+3, nsize+3);
            A.block(0,0,nsize,nsize) = K;
            A.block(0,nsize,nsize,3) = P;
            A.block(nsize,0,3,nsize) = P.transpose();

            VectorXd b = VectorXd::Zero(nsize+3);
            b.head(nsize) = T;

            VectorXd x = A.fullPivHouseholderQr().solve(b);

            Vector3d a = x.tail(3);
            VectorXd w = x.head(nsize);

            double sum = 0.0;
            double xi2 = xx(row, col);
            double yi2 = yy(row, col);
            for (auto j = 0; j < nsize; ++j)
            {
                double xj = Xn(j);
                double yj = Yn(j);
                double zj = Hn(j);
                if (std::isnan(xj) || std::isnan(yj) || std::isnan(zj))
                    continue;
                double rsqr = (xj - xi2) * (xj - xi2) + (yj - yi2) * (yj - yi2);
                if (rsqr == 0.0)
                    continue;
                sum += w(j) * rsqr * std::log10(std::sqrt(rsqr));
            }

            S(row, col) = a(0) + a(1)*xi2 + a(2)*yi2 + sum;
        }
    }

    return S;
}

Eigen::MatrixXd createDSM(PointView& view, int rows, int cols, double cell_size,
                          BOX2D bounds)
{
    using namespace Dimension;
    using namespace Eigen;

    MatrixXd ZImin(rows, cols);
    ZImin.setConstant(std::numeric_limits<double>::quiet_NaN());

    for (PointId i = 0; i < view.size(); ++i)
    {
        double x = view.getFieldAs<double>(Id::X, i);
        double y = view.getFieldAs<double>(Id::Y, i);
        double z = view.getFieldAs<double>(Id::Z, i);

        int c = Utils::clamp(static_cast<int>(floor(x-bounds.minx)/cell_size), 0, cols-1);
        int r = Utils::clamp(static_cast<int>(floor(y-bounds.miny)/cell_size), 0, rows-1);

        if (z < ZImin(r, c) || std::isnan(ZImin(r, c)))
            ZImin(r, c) = z;
    }

    return ZImin;
}

Eigen::MatrixXd matrixClose(Eigen::MatrixXd data, int radius)
{
    using namespace Eigen;

    MatrixXd data2 = padMatrix(data, radius);

    int nrows = data2.rows();
    int ncols = data2.cols();

    MatrixXd minZ = MatrixXd::Constant(nrows, ncols,
                                       std::numeric_limits<double>::max());
    MatrixXd maxZ = MatrixXd::Constant(nrows, ncols,
                                       std::numeric_limits<double>::lowest());
    for (auto c = 0; c < ncols; ++c)
    {
        int cs = Utils::clamp(c-radius, 0, ncols-1);
        int ce = Utils::clamp(c+radius, 0, ncols-1);
      
        for (auto r = 0; r < nrows; ++r)
        {
            int rs = Utils::clamp(r-radius, 0, nrows-1);
            int re = Utils::clamp(r+radius, 0, nrows-1);

            for (auto col = cs; col <= ce; ++col)
            {
                for (auto row = rs; row <= re; ++row)
                {
                    if ((row-r)*(row-r)+(col-c)*(col-c) > radius*radius)
                        continue;
                    if (data2(row, col) > maxZ(r, c))
                        maxZ(r, c) = data2(row, col);
                }
            }
        }
    }
    for (auto c = 0; c < ncols; ++c)
    {
        int cs = Utils::clamp(c-radius, 0, ncols-1);
        int ce = Utils::clamp(c+radius, 0, ncols-1);
      
        for (auto r = 0; r < nrows; ++r)
        {
            int rs = Utils::clamp(r-radius, 0, nrows-1);
            int re = Utils::clamp(r+radius, 0, nrows-1);

            for (auto col = cs; col <= ce; ++col)
            {
                for (auto row = rs; row <= re; ++row)
                {
                    if ((row-r)*(row-r)+(col-c)*(col-c) > radius*radius)
                        continue;
                    if (maxZ(row, col) < minZ(r, c))
                        minZ(r, c) = maxZ(row, col);
                }
            }
        }
    }

    return minZ.block(radius, radius, data.rows(), data.cols());
}

Eigen::MatrixXd matrixOpen(Eigen::MatrixXd data, int radius)
{
    using namespace Eigen;

    MatrixXd data2 = padMatrix(data, radius);

    int nrows = data2.rows();
    int ncols = data2.cols();

    MatrixXd minZ = MatrixXd::Constant(nrows, ncols,
                                       std::numeric_limits<double>::max());
    MatrixXd maxZ = MatrixXd::Constant(nrows, ncols,
                                       std::numeric_limits<double>::lowest());
    for (auto c = 0; c < ncols; ++c)
    {
        int cs = Utils::clamp(c-radius, 0, ncols-1);
        int ce = Utils::clamp(c+radius, 0, ncols-1);
      
        for (auto r = 0; r < nrows; ++r)
        {
            int rs = Utils::clamp(r-radius, 0, nrows-1);
            int re = Utils::clamp(r+radius, 0, nrows-1);

            for (auto col = cs; col <= ce; ++col)
            {
                for (auto row = rs; row <= re; ++row)
                {
                    if ((row-r)*(row-r)+(col-c)*(col-c) > radius*radius)
                        continue;
                    if (data2(row, col) < minZ(r, c))
                        minZ(r, c) = data2(row, col);
                }
            }
        }
    }
    for (auto c = 0; c < ncols; ++c)
    {
        int cs = Utils::clamp(c-radius, 0, ncols-1);
        int ce = Utils::clamp(c+radius, 0, ncols-1);
      
        for (auto r = 0; r < nrows; ++r)
        {
            int rs = Utils::clamp(r-radius, 0, nrows-1);
            int re = Utils::clamp(r+radius, 0, nrows-1);

            for (auto col = cs; col <= ce; ++col)
            {
                for (auto row = rs; row <= re; ++row)
                {
                    if ((row-r)*(row-r)+(col-c)*(col-c) > radius*radius)
                        continue;
                    if (minZ(row, col) > maxZ(r, c))
                        maxZ(r, c) = minZ(row, col);
                }
            }
        }
    }

    return maxZ.block(radius, radius, data.rows(), data.cols());
}

Eigen::MatrixXd padMatrix(Eigen::MatrixXd d, int r)
{
    using namespace Eigen;

    MatrixXd out = MatrixXd::Zero(d.rows()+2*r, d.cols()+2*r);
    out.block(r, r, d.rows(), d.cols()) = d;
    out.block(r, 0, d.rows(), r) =
        d.block(0, 0, d.rows(), r).rowwise().reverse();
    out.block(r, d.cols()+r, d.rows(), r) =
        d.block(0, d.cols()-r, d.rows(), r).rowwise().reverse();
    out.block(0, 0, r, out.cols()) =
        out.block(r, 0, r, out.cols()).colwise().reverse();
    out.block(d.rows()+r, 0, r, out.cols()) =
        out.block(out.rows()-r, 0, r, out.cols()).colwise().reverse();

    return out;
}

PDAL_DLL Eigen::MatrixXd pointViewToEigen(const PointView& view)
{
    Eigen::MatrixXd matrix(view.size(), 3);
    for (PointId i = 0; i < view.size(); ++i)
    {
        matrix(i, 0) = view.getFieldAs<double>(Dimension::Id::X, i);
        matrix(i, 1) = view.getFieldAs<double>(Dimension::Id::Y, i);
        matrix(i, 2) = view.getFieldAs<double>(Dimension::Id::Z, i);
    }
    return matrix;
}

} // namespace eigen

} // namespace pdal
