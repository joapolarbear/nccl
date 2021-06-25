# NCCL Profiler

## Usage

### ENV Setting
1. Set `NCCL_ENABLE_TIMELINE=1` to enable NCCL profiler.
2. Use `NCCL_TRACE_START_STEP` and `NCCL_TRACE_END_STEP` to decide the profiling range.
3. Set `NCCL_TRACE_DIR` to the path to store the traces 

### API
To support profiling a specific range of steps, a `unique_name` must be given when involking the collective operations. E.g. Involking of `ncclAllReduce` should be modified from
``` C++
ncclAllReduce(sendbuff, recvbuff, count, datatype, op, commstream)
```
to
``` C++
ncclAllReduce(sendbuff, recvbuff, count, datatype, op, commstream, tensor_name.c_str())
```
Currently, we only support tensor names in the format of `id<<step_id>>`, e.g. `1<<2>>` represents tensor `1` in step `2`. Here `id` should be able to be converted to an integer, or no traces will be outputed.
See [src/debug.cc#L353](src/debug.cc#L353) for more details.
