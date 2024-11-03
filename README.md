# IRGPUA Project

This project implements a GPU-accelerated image processing pipeline using CUDA and OpenMP. Follow the instructions below to build and run the code.

## Build Instructions

### Create a Build Directory

```bash
$ mkdir build
$ cd build/
```

### Run CMake

```bash
$ cmake ..
```

### Compile the Code

```bash
$ make
```

## Binaries

After a successful build, the following binaries will be produced in the `build` directory:

- **main_cpu**: The executable for the CPU implementation of the image processing pipeline.
- **main_gpu**: The executable for the standard GPU implementation.
- **main_gpu_indus**: The executable for the industrial GPU implementation.

## Running the Binaries

You can run each binary as follows:

```bash
$ ./main_cpu
$ ./main_gpu
$ ./main_gpu_indus
```