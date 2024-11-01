#pragma once

#ifndef _IRGPUA_CPU
#  include <raft/common/nvtx.hpp>
#  define WRAP_NVTX(Name, Cmd)                                                 \
    raft::common::nvtx::push_range(Name);                                      \
    Cmd;                                                                       \
    raft::common::nvtx::pop_range();
#endif
