#pragma once

#include "common.cuh"

struct ggml_cuda_concurrent_event {
    std::vector<cudaEvent_t> join_events;
    cudaEvent_t              fork_event = nullptr;

    int                                          n_streams = 0;
    std::unordered_map<const ggml_tensor *, int> stream_mapping;

    // Original order of nodes in this concurrent region (before interleaving)
    // Used to restore grouping for fusion within streams
    std::vector<const ggml_tensor *> original_order;

    const ggml_tensor * join_node;

    ggml_cuda_concurrent_event() = default;

    ggml_cuda_concurrent_event(const ggml_cuda_concurrent_event &) = delete;
    ggml_cuda_concurrent_event & operator=(const ggml_cuda_concurrent_event &) = delete;

    explicit ggml_cuda_concurrent_event(int n_streams);

    ggml_cuda_concurrent_event(ggml_cuda_concurrent_event && other) noexcept;

    // 1. check if any branches write to overlapping memory ranges (except the join node)
    // 2. check whether all srcs are either within the branch or outside the nodes covered by ggml_cuda_concurrent_event
    // we assume all nodes have the same buffer
    bool is_valid() const;

    ~ggml_cuda_concurrent_event();
};

struct ggml_cuda_stream_context {
    std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> concurrent_events;

    void reset();
};
