cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(project_irgpua LANGUAGES CXX CUDA)

set(RAFT_NVTX ON)
include(fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(tp_reduce)

rapids_cpm_init()
include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_raft.cmake)

find_package(OpenMP REQUIRED)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
set(CUDA_ARCHITECTURES "75")

include_directories(src)

# Build CPU version
# _IRGPUA_CPU
# add_executable(main_cpu src/main.cu src/fix_cpu.cu)
# target_compile_definitions(main_cpu PRIVATE _IRGPUA_CPU)
# target_compile_options(main_cpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
# target_link_libraries(main_cpu PUBLIC OpenMP::OpenMP_CXX)

# Build GPU version
# _IRGPUA_GPU
add_executable(main_gpu
  src/main.cu
  src/fix_cpu.cu
  src/scan.cu
  src/histogram.cu
  src/map_fix.cu
  src/compact.cu
  src/fix_gpu.cu
  src/cuda_tools/memory_pool.cu
  src/reduce.cu)
target_compile_definitions(main_gpu PRIVATE _IRGPUA_GPU)
target_compile_options(main_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(main_gpu PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)

# Build GPU INDUS version
# _IRGPUA_GPU_INDUS
# add_executable(main_gpu_indus
#   src/main.cu
#   src/fix_cpu.cu
#   src/scan.cu
#   src/histogram.cu
#   src/map_fix.cu
#   src/compact.cu
#   src/fix_gpu.cu
#   src/cuda_tools/memory_pool.cu
#   src/reduce.cu)
# target_compile_definitions(main_gpu_indus PRIVATE _IRGPUA_GPU_INDUS)
# target_compile_options(main_gpu_indus PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
# target_link_libraries(main_gpu_indus PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)
