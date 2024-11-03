#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _IRGPUA_CPU
#  include "cuda_tools/memory_pool.cuh"
#endif

struct Image
{
  Image() = default;

  Image(const std::string& filepath, int id = -1)
  {
    to_sort.id = id;

    std::ifstream infile(filepath, std::ifstream::binary);

    if (!infile.is_open())
      throw std::runtime_error("Failed to open");

    std::string magic;
    infile >> magic;
    infile.seekg(1, infile.cur);
    char c;
    infile.get(c);
    while (c == '#')
      {
        while (c != '\n')
          infile.get(c);
        infile.get(c);
      }

    infile.seekg(-1, infile.cur);

    int max;
    infile >> width >> height >> max;
    if (max != 255 && magic == "P5")
      throw std::runtime_error("Bad max value");

    if (magic == "P5")
      {
        actual_size = width * height;
#ifndef _IRGPUA_CPU
        buffer = static_cast<int*>(allocate_host_pinned_memory(
          actual_size * sizeof(int) + sizeof(uint64_t)));
        to_sort.total = reinterpret_cast<uint64_t*>(buffer + actual_size);
        *(to_sort.total) = 0;
#else
        buffer = (int*)malloc(width * height * sizeof(int));
#endif

        infile.seekg(1, infile.cur);
        for (int i = 0; i < actual_size; ++i)
          {
            uint8_t pixel_char;
            infile >> std::noskipws >> pixel_char;
            buffer[i] = pixel_char;
          }
      }
    else if (magic == "P?")
      {
        infile.seekg(1, infile.cur);

        std::string line;
        std::getline(infile, line);

        int image_size = 0;
        {
          std::stringstream lineStream(line);
          std::string s;

          while (std::getline(lineStream, s, ';'))
            ++image_size;
        }
#ifndef _IRGPUA_CPU
        buffer = static_cast<int*>(allocate_host_pinned_memory(
          image_size * sizeof(int) + sizeof(uint64_t)));
        to_sort.total = reinterpret_cast<uint64_t*>(buffer + image_size);
        *(to_sort.total) = 0;
#else
        buffer = (int*)malloc(image_size * sizeof(int));
#endif

        std::stringstream lineStream(line);
        std::string s;

        int i = 0;

        while (std::getline(lineStream, s, ';'))
          buffer[i++] = std::stoi(s);
        actual_size = i;
      }
    else
      throw std::runtime_error("Bad PPM value");
  }

  int size() const { return actual_size; }

  void write(const std::string& filepath) const
  {
    std::ofstream outfile(filepath, std::ofstream::binary);
    if (outfile.fail())
      throw std::runtime_error("Failed to open");
    outfile << "P5"
            << "\n"
            << width << " " << height << "\n"
            << 255 << "\n";

    for (int i = 0; i < height * width; ++i)
      {
        int val = buffer[i];
        if (val < 0 || val > 255)
          {
            std::cout << std::endl;
            std::cout << "Error at : " << i << " Value is : " << val
                      << ". Values should be between 0 and 255." << std::endl;
            throw std::runtime_error("Invalid image format");
          }
        outfile << static_cast<uint8_t>(val);
      }
  }

  int* buffer;
  int height = -1;
  int width = -1;
  int actual_size = -1;
  struct ToSort
  {
#ifndef _IRGPUA_CPU
    uint64_t* total = nullptr;
#else
    uint64_t total = 0;
#endif
    int id = -1;
  } to_sort;
};