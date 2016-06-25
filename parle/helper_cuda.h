#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_timer.h"

#include <stdio.h>

void CheckCudaError(cudaError ret, const char* stmt, const char* filename, int line)
{
    if (ret != cudaSuccess)
    {
        printf("CL error: \"%s\",at %s:%i\nfor statement\n\"%s\"\n", cudaGetErrorString(ret), filename, line, stmt);
        exit(1);
    }
}
#define _DEBUG

// GL Check.
#ifdef _DEBUG
#define CUDA_CHECK(stmt) do { \
            cudaError ret = stmt; \
            CheckCudaError(ret, #stmt, __FILE__, __LINE__);	\
        } while (0)
#else
#define CUDA_CHECK(stmt) stmt
#endif