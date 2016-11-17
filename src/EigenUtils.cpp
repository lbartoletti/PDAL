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

#include "gdal_priv.h" // For File I/O
#include "gdal_version.h" // For version info
#include "ogr_spatialref.h"  //For Geographic Information/Transformations

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

Eigen::MatrixXd extendedLocalMinimum(PointView& view, int rows, int cols,
                                     double cell_size, BOX2D bounds)
{
  using namespace Dimension;
  using namespace Eigen;

  // Index elevation values by row and column.
  std::map<uint32_t, std::vector<double>> hash;
  for (PointId i = 0; i < view.size(); ++i)
  {
      double x = view.getFieldAs<double>(Id::X, i);
      double y = view.getFieldAs<double>(Id::Y, i);
      double z = view.getFieldAs<double>(Id::Z, i);

      int c = Utils::clamp(static_cast<int>(floor(x-bounds.minx)/cell_size), 0, cols-1);
      int r = Utils::clamp(static_cast<int>(floor(y-bounds.miny)/cell_size), 0, rows-1);
      
      hash[r*cols+c].push_back(z);
  }
  
  // For each grid cell, sort elevations and detect local minimum, rejecting low
  // outliers.
  MatrixXd ZImin(rows, cols);
  ZImin.setConstant(std::numeric_limits<double>::quiet_NaN());
  for (int c = 0; c < cols; ++c)
  {
      for (int r = 0; r < rows; ++r)
      {
          std::vector<double> cp(hash[r*cols+c]);
          if (cp.empty())
              continue;
          std::sort(cp.begin(), cp.end());
          if (cp.size() == 1)
          {
              ZImin(r, c) = cp[0];
              continue;
          }
          for (size_t i = 0; i < cp.size()-1; ++i)
          {
              if (std::fabs(cp[i] - cp[i+1]) < 1.0)
              {
                  ZImin(r, c) = cp[i];
                  break;
              }
          }
      }
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

void writeMatrix(Eigen::MatrixXd data, std::string filename, double cell_size,
                 PointViewPtr view, BOX2D bounds)
{
    int cols = data.cols();
    int rows = data.rows();

    GDALAllRegister();

    GDALDataset *mpDstDS = NULL;

    char **papszMetadata;

    // parse the format driver, hardcoded for the time being
    std::string tFormat("GTIFF");
    const char *pszFormat = tFormat.c_str();
    GDALDriver* tpDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);

    // try to create a file of the requested format
    if (tpDriver != NULL)
    {
        papszMetadata = tpDriver->GetMetadata();
        if (CSLFetchBoolean(papszMetadata, GDAL_DCAP_CREATE, FALSE))
        {
            char **papszOptions = NULL;

            mpDstDS = tpDriver->Create(filename.c_str(), cols, rows, 1,
                                       GDT_Float32, papszOptions);

            // set the geo transformation
            double adfGeoTransform[6];
            adfGeoTransform[0] = bounds.minx;
            adfGeoTransform[1] = cell_size;
            adfGeoTransform[2] = 0.0;
            adfGeoTransform[3] = bounds.miny;
            adfGeoTransform[4] = 0.0;
            adfGeoTransform[5] = cell_size;
            mpDstDS->SetGeoTransform(adfGeoTransform);

            // set the projection
            mpDstDS->SetProjection(view->spatialReference().getWKT().c_str());
        }
    }

    // if we have a valid file
    if (mpDstDS)
    {
        // loop over the raster and determine max slope at each location
        int cs = 1, ce = cols - 1;
        int rs = 1, re = rows - 1;
        float *poRasterData = new float[cols*rows];
        for (auto i=0; i<cols*rows; i++)
        {
            poRasterData[i] = std::numeric_limits<float>::min();
        }

        // #pragma omp parallel for
        for (auto c = cs; c < ce; ++c)
        {
            for (auto r = rs; r < re; ++r)
            {
                if (data(r, c) == 0.0 || std::isnan(data(r, c)) || data(r, c) == std::numeric_limits<double>::max())
                    continue;
                poRasterData[(r * cols) + c] =
                    data(r, c);
            }
        }

        // write the data
        if (poRasterData)
        {
            GDALRasterBand *tBand = mpDstDS->GetRasterBand(1);

            tBand->SetNoDataValue(std::numeric_limits<float>::min());

            if (cols > 0 && rows > 0)
#if GDAL_VERSION_MAJOR <= 1
                tBand->RasterIO(GF_Write, 0, 0, cols, rows,
                                poRasterData, cols, rows,
                                GDT_Float32, 0, 0);
#else

                int ret = tBand->RasterIO(GF_Write, 0, 0, cols, rows,
                                          poRasterData, cols, rows,
                                          GDT_Float32, 0, 0, 0);
#endif
        }

        GDALClose((GDALDatasetH) mpDstDS);

        delete [] poRasterData;
    }
}

} // namespace eigen

} // namespace pdal
