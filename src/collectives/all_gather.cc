/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include <stdarg.h>

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, ...);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, ...) {

  // byteprofile, retrive the unique name
  const char* unique_name;
  va_list vargs;
  va_start(vargs, stream);
  const char* input_name = va_arg(vargs, const char *);
  va_end(vargs);
  if (input_name != NULL && strstr(input_name, "horovod") != NULL) unique_name = input_name;
  else unique_name = NULL;

  struct ncclInfo info = { ncclCollAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, unique_name, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
