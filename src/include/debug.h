/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEBUG_H_
#define NCCL_DEBUG_H_

#include "core.h"

#include <stdio.h>
#include <chrono>

#include <sys/syscall.h>
#include <limits.h>
#include <string.h>
#include "nccl_net.h"
#include <unordered_map>
#include <string>

#define gettid() (pid_t) syscall(SYS_gettid)

extern int ncclDebugLevel;
extern uint64_t ncclDebugMask;
extern pthread_mutex_t ncclDebugOutputLock;
extern FILE *ncclDebugFile;
extern ncclResult_t getHostName(char* hostname, int maxlen, const char delim);

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...);

typedef struct ncclSliceInfoT {
  int channelId;
  int chunkId;
  int sliceId;
} ncclSliceInfo;

int ncclAddTrace(const char *name, int rank, int local_rank, bool mark, long long start_t, ncclSliceInfo *sliceInfo);
void ncclOutputTrace();
void ncclGetCurTime(long long *ret);
bool isBPF_ON(int rank);

#define ENABLE_TRACE

#define MAX_TRACE_NAME_LEN 128
typedef struct ncclTraceT {
  char name[MAX_TRACE_NAME_LEN];
  char pid[MAX_TRACE_NAME_LEN];
  char tid[MAX_TRACE_NAME_LEN];
  long long ts = 0;
  long long dur = 0;
  char ph;

  int channelId;
  int chunkId;
  int sliceId;

  struct ncclTraceT* prev = NULL;
  struct ncclTraceT* next = NULL;
} ncclTrace;

struct pair_uint64_t_bool {
    uint64_t cnt;
    bool end;
};

// Let code temporarily downgrade WARN into INFO
extern thread_local int ncclDebugNoWarn;
#define NOWARN(a, ret) do { \
  ncclDebugNoWarn = 1; \
  ret = a; \
  ncclDebugNoWarn = 0; \
} while (0)

#define WARN(...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) ncclDebugLog(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

// for byteprofile
#define BPF_TRACE(...) ncclDebugLog(NCCL_LOG_BPF_TRACE, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__) 
#define BPF_TIMELINE(...) ncclAddTrace(__VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) ncclDebugLog(NCCL_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
extern std::chrono::high_resolution_clock::time_point ncclEpoch;
#else
#define TRACE(...)
#endif

#endif
