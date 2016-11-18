/******************************************************************************
* Copyright (c) 2016, Bradley J Chambers, brad.chambers@gmail.com
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

#include "SlopeWriter.hpp"

#include <pdal/EigenUtils.hpp>
#include <pdal/PointView.hpp>
#include <pdal/pdal_macros.hpp>

namespace pdal
{
static PluginInfo const s_info =
    PluginInfo("writers.slope", "Slope writer",
               "http://pdal.io/stages/writers.slope.html");

CREATE_STATIC_PLUGIN(1, 0, SlopeWriter, Writer, s_info)

std::string SlopeWriter::getName() const
{
    return s_info.name;
}

void SlopeWriter::addArgs(ProgramArgs& args)
{
    args.add("filename", "Output filename", m_filename).setPositional();
    args.add("cell_size", "Resolution/cell size", m_cellSize, 15.0);
}

void SlopeWriter::write(const PointViewPtr view)
{
    using namespace Eigen;
    
    // Bounds are required for computing number of rows and columns, and for
    // later indexing individual points into the appropriate raster cells.
    BOX2D bounds;
    view->calculateBounds(bounds);
    
    // Determine the number of rows and columns at the given cell size.
    size_t cols = ((bounds.maxx - bounds.minx) / m_cellSize) + 1;
    size_t rows = ((bounds.maxy - bounds.miny) / m_cellSize) + 1;
    
    MatrixXd DSM, cleanedDSM, slope;
    
    // Begin by creating a DSM of max elevations per XY cell.
    DSM = eigen::createMaxMatrix(*view.get(), rows, cols, m_cellSize, bounds);
    
    // Continue by cleaning the DSM (currently a noop).
    cleanedDSM = eigen::cleanDSM(DSM);
    
    // The actual slope calculation occurs here.
    slope = eigen::computeSlope(cleanedDSM, m_cellSize);
    
    // Finally, write our result as a raster (hardcoded as a GTiff for now).
    eigen::writeMatrix(slope, m_filename, m_cellSize, view, bounds);
}

} // namespace pdal
