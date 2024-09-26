#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "fix_cpu.cuh"
#include "fix_gpu.cuh"
#include "image.hh"
#include "pipeline.hh"
#include "cuda_tools/cuda_error_checking.cuh"

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
  // -- Pipeline initialization

  std::cout << "File loading..." << std::endl;

  // - Get file paths

  using recursive_directory_iterator =
    std::filesystem::recursive_directory_iterator;
  std::vector<std::string> filepaths;
  for (const auto& dir_entry : recursive_directory_iterator(
         "/afs/cri.epita.fr/resources/teach/IRGPUA/images"))
    filepaths.emplace_back(dir_entry.path());

  // - Init pipeline object

  Pipeline pipeline(filepaths);

  // -- Main loop containing image retring from pipeline and fixing

  const int nb_images = pipeline.images.size();
  std::vector<Image> images(nb_images);

  // - One CPU thread is launched for each image

  std::cout << "Done, starting compute" << std::endl;

  // #pragma omp parallel for
  for (int i = 0; i < nb_images; ++i)
    {
      // TODO : make it GPU compatible (aka faster)
      // You will need to copy images one by one on the GPU
      // You can store the images the way you want on the GPU
      // But you should treat the pipeline as a pipeline :
      // You *must not* copy all the images and only then do the computations
      // You must get the image from the pipeline as they arrive and launch computations right away
      // There are still ways to speeds this process of course
      images[i] = pipeline.get_image(i);
      fix_image_gpu(images[i]);
      // fix_image_cpu(images[i]);
    }

  std::cout << "Done with compute, starting stats" << std::endl;

// -- All images are now fixed : compute stats (total then sort)

// - First compute the total of each image

// TODO : make it GPU compatible (aka faster)
// You can use multiple CPU threads for your GPU version using openmp or not
// Up to you :)
  #pragma omp parallel for
for (int i = 0; i < nb_images; ++i)
{
    // Récupérer l'image
    auto& image = images[i];
    const size_t image_size = image.width * image.height;

    // Créer un flux CUDA spécifique au thread
    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));

    rmm::cuda_stream_view rmm_stream(stream);

    // Allouer de la mémoire sur le GPU avec RMM
    rmm::device_uvector<int> d_image(image_size, rmm_stream);

    // Copier l'image sur le GPU (asynchrone)
    CUDA_CHECK_ERROR(cudaMemcpyAsync(
        d_image.data(),
        image.buffer,
        image_size * sizeof(int),
        cudaMemcpyHostToDevice,
        rmm_stream.value()));

    // Synchroniser le flux pour s'assurer que la copie est terminée
    rmm_stream.synchronize();

    // Créer un device_ptr Thrust pour les opérations de réduction
    thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(d_image.data());

    // Calculer le total sur le GPU avec Thrust
    int total = thrust::reduce(
        thrust::cuda::par.on(rmm_stream.value()),
        dev_ptr,
        dev_ptr + image_size,
        0,
        thrust::plus<int>());

    // Synchroniser le flux pour s'assurer que la réduction est terminée
    rmm_stream.synchronize();

    // Stocker le total
    image.to_sort.total = total;

    // Détruire le flux CUDA
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream));
}

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
  std::sort(to_sort.begin(), to_sort.end(),
            [](ToSort a, ToSort b) { return a.total < b.total; });

  // TODO : Test here that you have the same results
  // You can compare visually and should compare image vectors values and "total" values
  // If you did the sorting, check that the ids are in the same order
  for (int i = 0; i < nb_images; ++i)
    {
      std::cout << "Image #" << images[i].to_sort.id
                << " total : " << images[i].to_sort.total << std::endl;
      std::ostringstream oss;
      oss << "Image#" << images[i].to_sort.id << ".pgm";
      std::string str = oss.str();
      images[i].write(str);
    }

  std::cout << "Done, the internet is safe now :)" << std::endl;

  // Cleaning
  // TODO : Don't forget to update this if you change allocation style
  for (int i = 0; i < nb_images; ++i)
    free(images[i].buffer);

  return 0;
}
