/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "nccl_net.h"
#include <stdlib.h>
#include <stdarg.h>

int ncclDebugLevel = -1;
thread_local int ncclDebugNoWarn = 0;
uint64_t ncclDebugMask = NCCL_INIT; // Default debug sub-system mask is INIT
FILE *ncclDebugFile = stdout;
pthread_mutex_t ncclDebugLock = PTHREAD_MUTEX_INITIALIZER;


// for byteprofile
int isTraceOn = -1;
int ncclByteProfileStart = -1, ncclByteProfileEnd = -1;
char ByteProfilePath[PATH_MAX+1] = "";
FILE *bpfFile = NULL;
ncclTrace* nccl_traces = NULL;
ncclTrace* nccl_last_trace = NULL;
std::unordered_map<std::string, struct pair_uint64_t_bool> trace_name_cnt;

int ncclParseFileName(const char *FileEnv, FILE **fd) {
  int c = 0;
  char debugFn[PATH_MAX+1] = "";
  char *dfn = debugFn;
  char hostname[1024];
  getHostName(hostname, 1024, '.');
  while (FileEnv[c] != '\0' && c < PATH_MAX) {
    if (FileEnv[c++] != '%') {
      *dfn++ = FileEnv[c-1];
      continue;
    }
    switch (FileEnv[c++]) {
      case '%': // Double %
        *dfn++ = '%';
        break;
      case 'h': // %h = hostname
        dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
        break;
      case 'p': // %p = pid
        dfn += snprintf(dfn, PATH_MAX, "%d", getpid());
        break;
      default: // Echo everything we don't understand
        *dfn++ = '%';
        *dfn++ = FileEnv[c-1];
        break;
    }
  }
  *dfn = '\0';
  if (debugFn[0] != '\0') {
    FILE *file = fopen(debugFn, "w");
    if (file != NULL) {
      printf("%s:%d DEBUG file is '%s'\n", hostname, getpid(), debugFn);
      *fd = file;
      return 0;
    }
  }
  return 1;
}

void ncclDebugInit() {
  pthread_mutex_lock(&ncclDebugLock);
  if (ncclDebugLevel != -1) return;
  const char* nccl_debug = getenv("NCCL_DEBUG");
  if (nccl_debug == NULL) {
    ncclDebugLevel = NCCL_LOG_NONE;
  } else if (strcasecmp(nccl_debug, "VERSION") == 0) {
    ncclDebugLevel = NCCL_LOG_VERSION;
  } else if (strcasecmp(nccl_debug, "WARN") == 0) {
    ncclDebugLevel = NCCL_LOG_WARN;
  } else if (strcasecmp(nccl_debug, "INFO") == 0) {
    ncclDebugLevel = NCCL_LOG_INFO;
  } else if (strcasecmp(nccl_debug, "ABORT") == 0) {
    ncclDebugLevel = NCCL_LOG_ABORT;
  } else if (strcasecmp(nccl_debug, "TRACE") == 0) {
    ncclDebugLevel = NCCL_LOG_TRACE;
  }

  /* Parse the NCCL_DEBUG_SUBSYS env var
   * This can be a comma separated list such as INIT,COLL
   * or ^INIT,COLL etc
   */
  char* ncclDebugSubsysEnv = getenv("NCCL_DEBUG_SUBSYS");
  if (ncclDebugSubsysEnv != NULL) {
    int invert = 0;
    if (ncclDebugSubsysEnv[0] == '^') { invert = 1; ncclDebugSubsysEnv++; }
    ncclDebugMask = invert ? ~0ULL : 0ULL;
    char *ncclDebugSubsys = strdup(ncclDebugSubsysEnv);
    char *subsys = strtok(ncclDebugSubsys, ",");
    while (subsys != NULL) {
      uint64_t mask = 0;
      if (strcasecmp(subsys, "INIT") == 0) {
        mask = NCCL_INIT;
      } else if (strcasecmp(subsys, "COLL") == 0) {
        mask = NCCL_COLL;
      } else if (strcasecmp(subsys, "P2P") == 0) {
        mask = NCCL_P2P;
      } else if (strcasecmp(subsys, "SHM") == 0) {
        mask = NCCL_SHM;
      } else if (strcasecmp(subsys, "NET") == 0) {
        mask = NCCL_NET;
      } else if (strcasecmp(subsys, "GRAPH") == 0) {
        mask = NCCL_GRAPH;
      } else if (strcasecmp(subsys, "TUNING") == 0) {
        mask = NCCL_TUNING;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = NCCL_ALL;
      }
      if (mask) {
        if (invert) ncclDebugMask &= ~mask; else ncclDebugMask |= mask;
      }
      subsys = strtok(NULL, ",");
    }
    free(ncclDebugSubsys);
  }

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NCCL_DEBUG level is > VERSION
   */
  const char* ncclDebugFileEnv = getenv("NCCL_DEBUG_FILE");
  if (ncclDebugLevel > NCCL_LOG_VERSION && ncclDebugFileEnv != NULL) {
    ncclParseFileName(ncclDebugFileEnv, &ncclDebugFile);
  }

#ifdef ENABLE_TRACE
  ncclEpoch = std::chrono::high_resolution_clock::now();
#endif

  pthread_mutex_unlock(&ncclDebugLock);
}

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) {
  if (ncclDebugLevel == -1) ncclDebugInit();
  if (ncclDebugNoWarn == 1 && level == NCCL_LOG_WARN) level = NCCL_LOG_INFO;

  char hostname[1024];
  getHostName(hostname, 1024, '.');
  int cudaDev;
  cudaGetDevice(&cudaDev);

  char buffer[1024];
  size_t len = 0;
  pthread_mutex_lock(&ncclDebugLock);
  if (ncclDebugNoWarn && ncclDebugLevel == NCCL_LOG_WARN) printf("WARN -> INFO\n");
  if (level == NCCL_LOG_WARN && ncclDebugLevel >= NCCL_LOG_WARN)
    len = snprintf(buffer, sizeof(buffer),
                   "\n%s:%d:%d [%d] %s:%d NCCL WARN ", hostname, getpid(), gettid(), cudaDev, filefunc, line);
  else if (level == NCCL_LOG_INFO && ncclDebugLevel >= NCCL_LOG_INFO && (flags & ncclDebugMask))
    len = snprintf(buffer, sizeof(buffer),
                   "%s:%d:%d [%d] NCCL INFO ", hostname, getpid(), gettid(), cudaDev);
#ifdef ENABLE_TRACE
  else if (level == NCCL_LOG_TRACE && ncclDebugLevel >= NCCL_LOG_TRACE && (flags & ncclDebugMask)) {
    auto delta = std::chrono::high_resolution_clock::now() - ncclEpoch;
    double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count()*1000;
    len = snprintf(buffer, sizeof(buffer),
                   "%s:%d:%d [%d] %f %s:%d NCCL TRACE ", hostname, getpid(), gettid(), cudaDev, timestamp, filefunc, line);
  }
#endif
  else if (level == NCCL_LOG_BPF_TRACE) {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    auto start_t = (long long)(us.count());
    len = snprintf(buffer, sizeof(buffer),
                   "%s:%d:%d [%d] %lld %s:%d NCCL BPF TRACE ", hostname, getpid(), gettid(), cudaDev, start_t, filefunc, line);
  }

  if (len) {
    va_list vargs;
    va_start(vargs, fmt);
    (void) vsnprintf(buffer+len, sizeof(buffer)-len, fmt, vargs);
    va_end(vargs);
    fprintf(ncclDebugFile,"%s\n", buffer);
    fflush(ncclDebugFile);
  }
  pthread_mutex_unlock(&ncclDebugLock);

  // If ncclDebugLevel == NCCL_LOG_ABORT then WARN() will also call abort()
  if (level == NCCL_LOG_WARN && ncclDebugLevel == NCCL_LOG_ABORT) {
    fprintf(stderr,"\n%s:%d:%d [%d] %s:%d NCCL ABORT\n",
            hostname, getpid(), gettid(), cudaDev, filefunc, line);
    abort();
  }
}

/** For byteprofile
 *
*/
void ncclTimelineInit(int local_rank) {
  pthread_mutex_lock(&ncclDebugLock);
  if (isTraceOn >= 0) {
    pthread_mutex_unlock(&ncclDebugLock);
    return; 
  }
  
  // for byteprofile
  char hostname[1024];
  getHostName(hostname, 1024, '.');
  const char* ncclByteProfileTrace = getenv("BYTEPS_TRACE_ON");
  if (ncclByteProfileTrace != NULL && ncclByteProfileTrace[0] == '1') {
    isTraceOn = 1;
    ncclByteProfileStart = std::stoi(getenv("BYTEPS_TRACE_START_STEP"));
    ncclByteProfileEnd = std::stoi(getenv("BYTEPS_TRACE_END_STEP"));
    printf("%s Timeline rang:[%d %d]\n", hostname, ncclByteProfileStart, ncclByteProfileEnd);

    const char* ncclByteProfileDir = getenv("BYTEPS_TRACE_DIR");
    snprintf(ByteProfilePath, sizeof(ByteProfilePath),
                   "%s/%d/comm_detail.json", ncclByteProfileDir, local_rank);
    printf("%s Timeline path: %s\n", hostname, ByteProfilePath);
    ncclParseFileName(ByteProfilePath, &bpfFile);
  } else {
    isTraceOn = 0;
    printf("%s BYTEPOS_TRACE_ON is not set\n", hostname);
  }
  pthread_mutex_unlock(&ncclDebugLock);
}

/** There may be some trival tensors which will only be transfered for
 * one or twcie (some small number). To avoid waiting for these tensors
 * to reach ncclByteProfileEnd, set a threashold, if the difference is 
 * too large, just ignore these tensors.
*/
bool ncclIsNeedArrive(int cnt) {
  if ((ncclByteProfileEnd - cnt) > (ncclByteProfileEnd - ncclByteProfileStart) / 2) {
    return false;
  } else {
    return true;
  }
}

void ncclGetCurTime(long long *ret) {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
  auto cur_t = (long long)(us.count());
  *ret = cur_t;
}

int ncclAddTrace(const char *name, int rank, int local_rank, bool mark, long long start_t){
  if (isTraceOn == -1) ncclTimelineInit(local_rank);
  if (bpfFile == NULL) return 0;

  pthread_mutex_lock(&ncclDebugLock);

  // Decide whether to output traces
  std::string name_str;
  if (name == NULL) {
    name_str = std::string("default_name");
    printf("%s: Input name is NULL\n", ByteProfilePath);
  } else {
    name_str = std::string(name);
  }
  std::unordered_map<std::string, struct pair_uint64_t_bool>::const_iterator finder = trace_name_cnt.find(name_str);
  if (finder == trace_name_cnt.end()) {
    trace_name_cnt[name_str] = {0, false};
    printf("%s ncclAddTrace adds new name %s\n", ByteProfilePath, name);
  } 
  if (trace_name_cnt[name_str].cnt >= ncclByteProfileEnd){
    if (bpfFile != NULL){
      // the first time larger than ncclByteProfileEnd
      trace_name_cnt[name_str].end = true;

      bool all_arrive = true;
      for (auto it = trace_name_cnt.begin(); it != trace_name_cnt.end(); ++ it) {
        if (!it->second.end && ncclIsNeedArrive((int)it->second.cnt)) {
          all_arrive = false;
          // printf("%s wait for %s\n", ByteProfilePath, it->first.c_str());
          break;
        }
      }
      if (all_arrive) {
        // all recorded tensors are ready to output
        ncclOutputTrace();
      }
    }
    pthread_mutex_unlock(&ncclDebugLock);
    return 0;
  } else if (mark){
    // for each slice, mark is false, we do not increase the tensor cnt
    // only when the tensor has been done, mark is set true
    trace_name_cnt[name_str].cnt += 1;
    pthread_mutex_unlock(&ncclDebugLock);
    return 0;
  }

  long long cur_t;
  ncclGetCurTime(&cur_t)

  ncclTrace *p_trace = (ncclTrace *)malloc(sizeof(ncclTrace));
  char debugFn[PATH_MAX+1];
  snprintf(debugFn, PATH_MAX, "comm_detail_r%d_lr%d", rank, local_rank);
  strcpy(p_trace->name, name);
  strcpy(p_trace->pid, debugFn);
  strcpy(p_trace->tid, "none");

  if (nccl_traces == NULL) {
    p_trace->ts = start_t;
    p_trace->dur = cur_t - p_trace->ts;
    p_trace->prev = NULL;
    p_trace->next = NULL;
    nccl_traces = p_trace;
    nccl_last_trace = p_trace;
  } else {
    auto last_ent_t = nccl_last_trace->ts + nccl_last_trace->dur
    p_trace->ts = (start_t > last_ent_t) ? start_t : last_ent_t;
    p_trace->dur = cur_t - p_trace->ts;
    p_trace->prev = nccl_last_trace;
    p_trace->next = NULL;
    nccl_last_trace->next = p_trace;
    nccl_last_trace = p_trace;
  }
  pthread_mutex_unlock(&ncclDebugLock);
  return 0;
}

void ncclOutputTrace() {
  ncclTrace *p_trace = nccl_traces;
  while (p_trace != NULL) {
    if (p_trace->prev == NULL){
      // the first trace
      fprintf(bpfFile, "{\n    \"traceEvents\": [\n");
    } else {
      fprintf(bpfFile, ",\n");
    }

    fprintf(bpfFile,
          "        {\n"
          "            \"ph\": \"X\",\n"
          "            \"args\": {\n"
          "                \"name\": \"%s\"\n"
          "            },\n"
          "            \"pid\": \"%s\",\n"
          "            \"name\": \"%s\",\n"
          "            \"ts\": %lld,\n"
          "            \"dur\": %lld,\n"
          "            \"tid\": \"%s\",\n"
          "            \"cat\": \"Comm\"\n"
          "        }", 
          p_trace->name, 
          p_trace->pid, 
          p_trace->name, 
          p_trace->ts, 
          p_trace->dur, 
          p_trace->tid);
    fflush(bpfFile);
    p_trace = p_trace->next;
  }
  fprintf(bpfFile, "\n"
     "    ],\n"
     "    \"displayTimeUnit\": \"ms\"\n"
     "}\n");
  fflush(bpfFile);
  fclose(bpfFile);
  bpfFile = NULL;
  printf("byteprofile output nccl trace to %s\n", ByteProfilePath);
}

bool isBPF_ON(int local_rank) {
  if (isTraceOn == -1) ncclTimelineInit(local_rank);
  return isTraceOn == 1;
}

