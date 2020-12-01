/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include <stdarg.h>

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, ...);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, ...) {
  
  // byteprofile, retrive the unique name
  const char* unique_name;
  va_list vargs;
  va_start(vargs, stream);
  const char* input_name = va_arg(vargs, const char *);
  va_end(vargs);
  if (input_name != NULL) unique_name = input_name;
  else unique_name = NULL;
  
  struct ncclInfo info = { ncclCollReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, unique_name, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
