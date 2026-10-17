#include "stream-context.cuh"

ggml_cuda_stream_context & ggml_backend_cuda_context::stream_context() {
    if (!concurrent_stream_context) {
        concurrent_stream_context = std::make_unique<ggml_cuda_stream_context>();
    }
    return *concurrent_stream_context;
}

ggml_cuda_concurrent_event::ggml_cuda_concurrent_event(int n_streams) : n_streams(n_streams) {
    join_events.resize(n_streams);

    for (size_t i = 0; i < join_events.size(); ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&join_events[i], cudaEventDisableTiming));
    }

    CUDA_CHECK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
}

ggml_cuda_concurrent_event::ggml_cuda_concurrent_event(ggml_cuda_concurrent_event && other) noexcept
: join_events(std::move(other.join_events))
, fork_event(other.fork_event)
, n_streams(other.n_streams)
, stream_mapping(std::move(other.stream_mapping))
, original_order(std::move(other.original_order))
, join_node(other.join_node) {
    other.fork_event = nullptr;
}

ggml_cuda_concurrent_event::~ggml_cuda_concurrent_event() {
    if (fork_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(fork_event));
    }
    for (cudaEvent_t e : join_events) {
        if (e != nullptr) {
            CUDA_CHECK(cudaEventDestroy(e));
        }
    }
}

bool ggml_cuda_concurrent_event::is_valid() const {
    std::vector<std::vector<std::pair<int64_t, int64_t>>> write_ranges;
    write_ranges.resize(n_streams);

    // get join_node's memory range to exclude from overlap checking.
    // multiple nodes can use join_node's buffer; we synchronize on the join node.
    const ggml_tensor * join_t     = join_node->view_src ? join_node->view_src : join_node;
    const int64_t       join_start = (int64_t) join_t->data;
    const int64_t       join_end   = join_start + ggml_nbytes(join_t);

    for (const auto & [tensor, stream] : stream_mapping) {
        const ggml_tensor * t = tensor->view_src ? tensor->view_src : tensor;
        const int64_t       t_start = (int64_t) t->data;
        const int64_t       t_end   = t_start + ggml_nbytes(t);

        // skip tensors that overlap with join_node's buffer.
        if ((t_start <= join_start && join_start < t_end) || (join_start <= t_start && t_start < join_end)) {
            continue;
        }

        // concurrent streams begin from 1
        write_ranges[stream - 1].emplace_back(t_start, t_end);
    }

    for (int i = 0; i < n_streams; ++i) {
        // sorts first by start then by end of write range
        std::sort(write_ranges[i].begin(), write_ranges[i].end());
    }

    bool writes_overlap = false;
    bool dependent_srcs = false;
    for (const auto & [tensor, stream] : stream_mapping) {
        const ggml_tensor * t = tensor->view_src ? tensor->view_src : tensor;
        const int64_t       t_start = (int64_t) t->data;
        const int64_t       t_end   = t_start + ggml_nbytes(t);

        // skip tensors that overlap with join_node's buffer
        if ((t_start <= join_start && join_start < t_end) || (join_start <= t_start && t_start < join_end)) {
            continue;
        }

        // check if this buffer's write data overlaps with another stream's
        std::pair<int64_t, int64_t> data_range = std::make_pair(t_start, t_end);
        for (int i = 0; i < n_streams; ++i) {
            if (i == stream - 1) {
                continue;
            }
            auto it = std::lower_bound(write_ranges[i].begin(), write_ranges[i].end(), data_range);

            if (it != write_ranges[i].end()) {
                const std::pair<int64_t, int64_t> & other = *it;

                // std::lower_bound returns the first element where other >= data_range (lexicographically).
                // This guarantees other.first >= data_range.first.
                // Therefore, overlap occurs iff other.first < data_range.second
                // (i.e., the other range starts before this range ends).
                if (other.first < data_range.second) {
                    GGML_LOG_DEBUG("Writes overlap for %s", tensor->name);
                    writes_overlap = true;
                    break;
                }
            }
        }

        //check if all srcs are either in branch or don't have a branch
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (!tensor->src[i]) {
                continue;
            }

            auto it = stream_mapping.find(tensor->src[i]);

            if (it == stream_mapping.end()) {
                continue;
            }

            if (it->second != stream) {
                dependent_srcs = true;
                break;
            }
        }

        if (dependent_srcs || writes_overlap) {
            break;
        }
    }

    return !writes_overlap && !dependent_srcs;
}

void ggml_cuda_stream_context::reset() {
    concurrent_events.clear();
}
