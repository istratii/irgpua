#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#ifndef _IRGPUA_CPU
#  include "cuda_tools/constants.hh"
#  include "cuda_tools/memory_pool.cuh"
#  include "fix_gpu.cuh"
#endif
#include "fix_cpu.cuh"
#include "image.hh"
#include "pipeline.hh"

#ifndef _IRGPUA_CPU
#  define HOST_PINNED_MEMORY_POOL_SIZE (192 * (1 << 20)) // 192 mega bytes
#  define DEVICE_MEMORY_POOL_SIZE (bytes_per_chunk * 30) // one gigabyte
#endif

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
  // -- Pipeline initialization

  std::cout << "File loading..." << std::endl;

  // - Get file paths

  using recursive_directory_iterator =
    std::filesystem::recursive_directory_iterator;
  std::vector<std::string> filepaths;
  for (const auto& dir_entry : recursive_directory_iterator(
         "/home/ucin/projects/epita/s9/irgpua/irgpua/images"))
    filepaths.emplace_back(dir_entry.path());

#ifndef _IRGPUA_CPU
  init_host_pinned_memory_pool(HOST_PINNED_MEMORY_POOL_SIZE);
  init_device_memory_pool(DEVICE_MEMORY_POOL_SIZE);
#endif

  // - Init pipeline object
  std::cout << "Done, starting compute" << '\n';

  Pipeline pipeline(filepaths);

  // -- Main loop containing image retring from pipeline and fixing

  const int nb_images = pipeline.images.size();
  std::vector<Image>& images(pipeline.images);

  // - One CPU thread is launched for each image

  std::cout << "Done with compute, starting stats" << '\n';

  // -- All images are now fixed : compute stats (total then sort)

#ifdef _IRGPUA_CPU
#  pragma omp parallel for
  for (int i = 0; i < nb_images; ++i)
    {
      auto& image = images[i];
      const int image_size = image.width * image.height;
      image.to_sort.total =
        std::reduce(image.buffer, image.buffer + image_size, 0);
    }
#endif

  // - All totals are known, sort images accordingly (OPTIONAL)
  // Moving the actual images is too expensive, sort image indices instead
  // Copying to an id array and sort it instead

  // TODO OPTIONAL : for you GPU version you can store it the way you want
  // But just like the CPU version, moving the actual images while sorting will be too slow
  using ToSort = Image::ToSort;
  std::vector<ToSort> to_sort(nb_images);
  std::generate(to_sort.begin(), to_sort.end(),
                [n = 0, images]() mutable { return images[n++].to_sort; });

  // TODO OPTIONAL : make it GPU compatible (aka faster)
  std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
#ifndef _IRGPUA_CPU
    return *(a.total) < *(b.total);
#else
    return a.total < b.total;
#endif
  });

#ifdef _IRGPUA_CPU
  for (int ii = 0; ii < nb_images; ++ii)
    {
      std::cout << "Image #" << images[ii].to_sort.id
                << " total : " << images[ii].to_sort.total << std::endl;
      std::ostringstream oss;
      oss << "Image#" << images[ii].to_sort.id << ".pgm";
      std::string str = oss.str();
      images[ii].write(str);
    }
#else
#  pragma omp parallel for
  for (int ii = 0; ii < nb_images; ++ii)
    {
      std::ostringstream oss;
      oss << "Image#" << images[ii].to_sort.id << ".pgm";
      images[ii].write(oss.str());
    }

  for (int ii = 0; ii < nb_images; ++ii)
    std::cout << "Image #" << images[ii].to_sort.id
              << " total : " << *(images[ii].to_sort.total) << '\n';
#endif

  std::cout << "Done, the internet is safe now :)" << '\n';

#ifndef _IRGPUA_CPU
  // Cleaning
  // DONE : Don't forget to update this if you change allocation style
  for (int i = 0; i < nb_images; ++i)
    {
      free_host_pinned_memory(images[i].to_sort.total, sizeof(uint64_t));
      free_host_pinned_memory(images[i].buffer, images[i].size() * sizeof(int));
    }

  free_host_pinned_memory_pool();
  free_device_memory_pool();
#else
  for (int ii = 0; ii < nb_images; ++ii)
    free(images[ii].buffer);
#endif

  return 0;
}
