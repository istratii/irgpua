#pragma once

#include <cstddef>
#include "alignement.hh"

constexpr size_t max_image_size = ALIGN128(4'016'251);
constexpr size_t bytes_per_image = max_image_size * sizeof(int);
constexpr size_t bytes_per_predicate = bytes_per_image;
#ifdef _IRGPUA_GPU
constexpr size_t bytes_per_scan = ALIGN128(sizeof(int) * (4096 * 3 + 1));
#else // _IRGPUA_GPU_INDUS
constexpr size_t bytes_per_temp_storage = 0; // ALIGN128(142'000);
#endif
constexpr size_t bytes_per_histogram = sizeof(int) * 256;
constexpr size_t bytes_per_cdf_min = sizeof(int);
constexpr size_t bytes_per_total = sizeof(int);
constexpr size_t bytes_per_chunk = bytes_per_image + bytes_per_predicate
  + bytes_per_histogram + bytes_per_cdf_min + bytes_per_total +
#ifdef _IRGPUA_GPU
  bytes_per_scan
#else // _IRGPUA_GPU_INDUS
  bytes_per_temp_storage
#endif
  ;
constexpr size_t buffer_offset = 0;
constexpr size_t predicate_offset = buffer_offset + bytes_per_image;
#ifdef _IRGPUA_GPU
constexpr size_t scan_offset =
#else // _IRGPUA_GPU_INDUS
constexpr size_t temp_storage_offset =
#endif
  predicate_offset + bytes_per_predicate;
constexpr size_t histogram_offset =
#ifdef _IRGPUA_GPU
  scan_offset + bytes_per_scan
#else // _IRGPUA_GPU_INDUS
  temp_storage_offset + bytes_per_temp_storage
#endif
  ;
constexpr size_t cdf_min_offset = histogram_offset + bytes_per_histogram;
constexpr size_t total_offset = cdf_min_offset + bytes_per_cdf_min;