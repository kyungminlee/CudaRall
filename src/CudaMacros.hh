#pragma once

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#else
#define CUDA_DEVICE
#define CUDA_HOST
#endif
