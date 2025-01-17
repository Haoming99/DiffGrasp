/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: kdtree_flann.h 36261 2011-02-26 01:34:42Z mariusm $
 *
 */

#pragma once

#include </home/haoming/Downloads/3DSGrasp-master/pcl/kdtree/include/pcl/kdtree/kdtree.h>
#include </usr/include/flann/util/params.h>

#include <memory>

// Forward declarations
namespace flann
{
  template <typename T> struct L2_Simple;
  template <typename T> class Index;
}

namespace pcl
{
namespace detail {
// Helper struct to create a compatible Matrix and copy data back when needed
// Replace using if constexpr in C++17
template <typename IndexT>
struct compat_with_flann : std::false_type {};

template <>
struct compat_with_flann<std::size_t> : std::true_type {};

template <typename IndexT>
using CompatWithFlann = std::enable_if_t<compat_with_flann<IndexT>::value, bool>;
template <typename IndexT>
using NotCompatWithFlann = std::enable_if_t<!compat_with_flann<IndexT>::value, bool>;
} // namespace detail

/**
 * @brief Compatibility template function to allow use of various types of indices with
 * FLANN
 * @details Template is used for all params to not constrain any FLANN side capability
 * @param[in,out] index A index searcher, of type ::flann::Index<Dist> or similar, where
 * Dist is a template for computing distance between 2 points
 * @param[in] query A ::flann::Matrix<float> or compatible matrix representation of the
 * query point
 * @param[out] indices Indices found in radius
 * @param[out] dists Computed distance matrix
 * @param[in] radius Threshold for consideration
 * @param[in] params Any parameters to pass to the radius_search call
 */
template <class FlannIndex,
          class Query,
          class Indices,
          class Distances,
          class SearchParams>
int
radius_search(const FlannIndex& index,
              const Query& query,
              Indices& indices,
              Distances& dists,
              float radius,
              const SearchParams& params);

/**
 * @brief Compatibility template function to allow use of various types of indices with
 * FLANN
 * @details Template is used for all params to not constrain any FLANN side capability
 * @param[in,out] index A index searcher, of type ::flann::Index<Dist> or similar, where
 * Dist is a template for computing distance between 2 points
 * @param[in] query A ::flann::Matrix<float> or compatible matrix representation of the
 * query point
 * @param[out] indices Neighboring k indices found
 * @param[out] dists Computed distance matrix
 * @param[in] k Number of neighbors to search for
 * @param[in] params Any parameters to pass to the knn_search call
 */
template <class FlannIndex,
          class Query,
          class Indices,
          class Distances,
          class SearchParams>
int
knn_search(const FlannIndex& index,
           const Query& query,
           Indices& indices,
           Distances& dists,
           unsigned int k,
           const SearchParams& params);

/** \brief KdTreeFLANN is a generic type of 3D spatial locator using kD-tree structures.
 * The class is making use of the FLANN (Fast Library for Approximate Nearest Neighbor)
 * project by Marius Muja and David Lowe.
 *
 * \author Radu B. Rusu, Marius Muja
 * \ingroup kdtree
 */
template <typename PointT, typename Dist = ::flann::L2_Simple<float>>
class KdTreeFLANN : public pcl::KdTree<PointT> {
public:
  using KdTree<PointT>::input_;
  using KdTree<PointT>::indices_;
  using KdTree<PointT>::epsilon_;
  using KdTree<PointT>::sorted_;
  using KdTree<PointT>::point_representation_;
  using KdTree<PointT>::nearestKSearch;
  using KdTree<PointT>::radiusSearch;

  using PointCloud = typename KdTree<PointT>::PointCloud;
  using PointCloudConstPtr = typename KdTree<PointT>::PointCloudConstPtr;

  using IndicesPtr = shared_ptr<Indices>;
  using IndicesConstPtr = shared_ptr<const Indices>;

  using FLANNIndex = ::flann::Index<Dist>;

  // Boost shared pointers
  using Ptr = shared_ptr<KdTreeFLANN<PointT, Dist>>;
  using ConstPtr = shared_ptr<const KdTreeFLANN<PointT, Dist>>;

  /** \brief Default Constructor for KdTreeFLANN.
   * \param[in] sorted set to true if the application that the tree will be used for
   * requires sorted nearest neighbor indices (default). False otherwise.
   *
   * By setting sorted to false, the \ref radiusSearch operations will be faster.
   */
  KdTreeFLANN(bool sorted = true);

  /** \brief Copy constructor
   * \param[in] k the tree to copy into this
   */
  KdTreeFLANN(const KdTreeFLANN<PointT, Dist>& k);

  /** \brief Copy operator
   * \param[in] k the tree to copy into this
   */
  inline KdTreeFLANN<PointT, Dist>&
  operator=(const KdTreeFLANN<PointT, Dist>& k)
  {
    KdTree<PointT>::operator=(k);
    flann_index_ = k.flann_index_;
    cloud_ = k.cloud_;
    index_mapping_ = k.index_mapping_;
    identity_mapping_ = k.identity_mapping_;
    dim_ = k.dim_;
    total_nr_points_ = k.total_nr_points_;
    param_k_ = k.param_k_;
    param_radius_ = k.param_radius_;
    return (*this);
  }

  /** \brief Set the search epsilon precision (error bound) for nearest neighbors
   * searches. \param[in] eps precision (error bound) for nearest neighbors searches
   */
  void
  setEpsilon(float eps) override;

  void
  setSortedResults(bool sorted);

  inline Ptr
  makeShared()
  {
    return Ptr(new KdTreeFLANN<PointT, Dist>(*this));
  }

  /** \brief Destructor for KdTreeFLANN.
   * Deletes all allocated data arrays and destroys the kd-tree structures.
   */
  ~KdTreeFLANN() override
  {
    cleanup();
  }

  /** \brief Provide a pointer to the input dataset.
   * \param[in] cloud the const boost shared pointer to a PointCloud message
   * \param[in] indices the point indices subset that is to be used from \a cloud - if
   * NULL the whole cloud is used
   */
  void
  setInputCloud(const PointCloudConstPtr& cloud,
                const IndicesConstPtr& indices = IndicesConstPtr()) override;

  /** \brief Search for k-nearest neighbors for the given query point.
   *
   * \attention This method does not do any bounds checking for the input index
   * (i.e., index >= cloud.size () || index < 0), and assumes valid (i.e., finite) data.
   *
   * \param[in] point a given \a valid (i.e., finite) query point
   * \param[in] k the number of neighbors to search for
   * \param[out] k_indices the resultant indices of the neighboring points (must be
   * resized to \a k a priori!) \param[out] k_sqr_distances the resultant squared
   * distances to the neighboring points (must be resized to \a k a priori!) \return
   * number of neighbors found
   *
   * \exception asserts in debug mode if the index is not between 0 and the maximum
   * number of points
   */
  int
  nearestKSearch(const PointT& point,
                 unsigned int k,
                 Indices& k_indices,
                 std::vector<float>& k_sqr_distances) const override;

  /** \brief Search for all the nearest neighbors of the query point in a given radius.
   *
   * \attention This method does not do any bounds checking for the input index
   * (i.e., index >= cloud.size () || index < 0), and assumes valid (i.e., finite) data.
   *
   * \param[in] point a given \a valid (i.e., finite) query point
   * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
   * \param[out] k_indices the resultant indices of the neighboring points
   * \param[out] k_sqr_distances the resultant squared distances to the neighboring
   * points \param[in] max_nn if given, bounds the maximum returned neighbors to this
   * value. If \a max_nn is set to 0 or to a number higher than the number of points in
   * the input cloud, all neighbors in \a radius will be returned. \return number of
   * neighbors found in radius
   *
   * \exception asserts in debug mode if the index is not between 0 and the maximum
   * number of points
   */
  int
  radiusSearch(const PointT& point,
               double radius,
               Indices& k_indices,
               std::vector<float>& k_sqr_distances,
               unsigned int max_nn = 0) const override;

private:
  /** \brief Internal cleanup method. */
  void
  cleanup();

  /** \brief Converts a PointCloud to the internal FLANN point array representation.
   * Returns the number of points. \param cloud the PointCloud
   */
  void
  convertCloudToArray(const PointCloud& cloud);

  /** \brief Converts a PointCloud with a given set of indices to the internal FLANN
   * point array representation. Returns the number of points. \param[in] cloud the
   * PointCloud data \param[in] indices the point cloud indices
   */
  void
  convertCloudToArray(const PointCloud& cloud, const Indices& indices);

private:
  /** \brief Class getName method. */
  std::string
  getName() const override
  {
    return ("KdTreeFLANN");
  }

  /** \brief A FLANN index object. */
  std::shared_ptr<FLANNIndex> flann_index_;

  /** \brief Internal pointer to data. TODO: replace with std::shared_ptr<float[]> with
   * C++17*/
  std::shared_ptr<float> cloud_;

  /** \brief mapping between internal and external indices. */
  std::vector<int> index_mapping_;

  /** \brief whether the mapping between internal and external indices is identity */
  bool identity_mapping_{false};

  /** \brief Tree dimensionality (i.e. the number of dimensions per point). */
  int dim_{0};

  /** \brief The total size of the data (either equal to the number of points in the
   * input cloud or to the number of indices - if passed). */
  uindex_t total_nr_points_{0};

  /** \brief The KdTree search parameters for K-nearest neighbors. */
  ::flann::SearchParams param_k_;

  /** \brief The KdTree search parameters for radius search. */
  ::flann::SearchParams param_radius_;
  };
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#endif
