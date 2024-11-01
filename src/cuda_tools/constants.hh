#pragma once

#include <cstddef>
#include "alignement.hh"

constexpr size_t max_image_size = ALIGN128(4'016'251);
constexpr size_t bytes_per_image = max_image_size * sizeof(int);
constexpr size_t bytes_per_predicate = bytes_per_image;
constexpr size_t bytes_per_scan = ALIGN128(sizeof(int) * (4096 * 3 + 1));
constexpr size_t bytes_per_histogram = sizeof(int) * 256;
constexpr size_t bytes_per_cdf_min = ALIGN128(sizeof(int));
constexpr size_t bytes_per_total = ALIGN128(sizeof(int));
constexpr size_t bytes_per_chunk = bytes_per_image + bytes_per_predicate
  + bytes_per_scan + bytes_per_histogram + bytes_per_cdf_min + bytes_per_total;
constexpr size_t buffer_offset = 0;
constexpr size_t predicate_offset = buffer_offset + bytes_per_image;
constexpr size_t scan_offset = predicate_offset + bytes_per_predicate;
constexpr size_t histogram_offset = scan_offset + bytes_per_scan;
constexpr size_t cdf_min_offset = histogram_offset + bytes_per_histogram;
constexpr size_t total_offset = cdf_min_offset + bytes_per_cdf_min;