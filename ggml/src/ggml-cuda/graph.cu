#include "graph.cuh"

#ifdef USE_CUDA_GRAPH

ggml_cuda_graph * ggml_cuda_graph_cache::get(const void * first_node_ptr) {
    const int64_t time_now = ggml_time_us();

    // sweep every 5s, evicting cuda graphs unused for >=10s
    if (time_now - last_graph_eviction_sweep >= 5'000'000) {
        last_graph_eviction_sweep = time_now;
        for (auto it = cuda_graphs.begin(); it != cuda_graphs.end(); ) {
            if (time_now - it->second->last_used_time >= 10'000'000) {
                it = cuda_graphs.erase(it);
            } else {
                ++it;
            }
        }
    }

    auto it = cuda_graphs.find(first_node_ptr);
    if (it == cuda_graphs.end()) {
        it = cuda_graphs.emplace(first_node_ptr, std::make_unique<ggml_cuda_graph>()).first;
    }
    it->second->last_used_time = time_now;
    return it->second.get();
}

ggml_cuda_graph * ggml_backend_cuda_context::cuda_graph(const void * first_node_ptr) {
    if (!graph_cache) {
        graph_cache = std::make_unique<ggml_cuda_graph_cache>();
    }
    return graph_cache->get(first_node_ptr);
}

bool ggml_backend_cuda_context::any_cuda_graph_enabled() const {
    if (graph_cache) {
        for (const auto & [key, graph] : graph_cache->cuda_graphs) {
            if (graph && graph->is_enabled()) {
                return true;
            }
        }
    }
    return false;
}

bool ggml_backend_cuda_context::any_cuda_graph_has_instance() const {
    if (graph_cache) {
        for (const auto & [key, graph] : graph_cache->cuda_graphs) {
            if (graph && graph->instance != nullptr) {
                return true;
            }
        }
    }
    return false;
}
#endif // USE_CUDA_GRAPH
