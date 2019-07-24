/** \file
 * \brief Provides memory storage tuned to a platform
 */

#ifndef BLASTED_DEVICE_CONTAINER_H
#define BLASTED_DEVICE_CONTAINER_H

#include <vector>
#include <boost/align/aligned_allocator.hpp>

/// Cache line length in bytes, used for aligned allocation on CPUs
#define CACHE_LINE_LEN 64

namespace blasted {

/// An aligned dynamic array for CPUs
/** Note that the template type must be a POD ('plain old data') type.
 */
template <typename T>
using device_vector = std::vector<T, boost::alignment::aligned_allocator<T,CACHE_LINE_LEN>>;

}
#endif
