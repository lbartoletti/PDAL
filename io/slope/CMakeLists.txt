PDAL_ADD_DRIVER(writer slope "SlopeWriter.cpp" "SlopeWriter.hpp" objects)
set(PDAL_TARGET_OBJECTS ${PDAL_TARGET_OBJECTS} ${objects} PARENT_SCOPE)
