#pragma once

#include "image.hh"

#include <memory>
#include <regex>
#include <stack>
#include <string>
#include <raft/core/handle.hpp>
#include <rmm/device_buffer.hpp>
#include "fix_gpu.cuh"

static std::string get_number(const std::string& str)
{
  std::regex r("Broken#(\\d+)");
  std::smatch match;

  if (std::regex_search(str, match, r) && match.size() > 1)
    return match.str(1);
  else
    {
      std::cerr << "Error file name" << std::endl;
      exit(-1);
      return "Error file name";
    }
}

struct Pipeline
{
  Pipeline(const std::vector<std::string>& filepaths)
  {
    const unsigned int N = filepaths.size();
    images = std::vector<Image>(N);
    handlers = std::vector<raft::handle_t>(N);

#pragma omp parallel for
    for (std::size_t ii = 0; ii < N; ++ii)
      {
        const int image_id = std::stoi(get_number(filepaths[ii]));
        images[ii] = Image(filepaths[ii], image_id);
        fix_image_gpu(images[ii], handlers[ii].get_stream());
      }

    CUDA_CHECK_ERROR2(cudaDeviceSynchronize(), "end");
  }

  Image&& get_image(int i) { return std::move(images[i]); }

  std::vector<Image> images;
  std::vector<raft::handle_t> handlers;
};
