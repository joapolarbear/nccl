/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include <stdarg.h>

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream, ...);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream, ...) {
  
  // byteprofile, retrive the unique name
  const char* unique_name;
  va_list vargs;
  va_start(vargs, stream);
  const char* input_name = va_arg(vargs, const char *);
  va_end(vargs);
  if (input_name != NULL && strstr(input_name, "horovod") != NULL) unique_name = input_name;
  else unique_name = NULL;

  struct ncclInfo info = { ncclCollBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, unique_name, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream, ...);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream, ...) {
  
  // byteprofile, retrive the unique name
  const char* unique_name;
  va_list vargs;
  va_start(vargs, stream);
  const char* input_name = va_arg(vargs, const char *);
  va_end(vargs);
  if (input_name != NULL && strstr(input_name, "horovod") != NULL) unique_name = input_name;
  else unique_name = NULL;

  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream, unique_name);
}

