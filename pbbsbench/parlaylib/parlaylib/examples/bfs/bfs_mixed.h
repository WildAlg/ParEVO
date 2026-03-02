#include <atomic>
#include <vector>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <omp.h>

template <typename vertex, typename graph>
auto BFS(vertex start, const graph& G) {
  using nested_seq = parlay::sequence<parlay::sequence<vertex>>;
  size_t n = G.size();
  
  // Atomic visited flags
  parlay::sequence<std::atomic<bool>> visited(n);
  #pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    visited[i].store(i == start, std::memory_order_relaxed);
  }


  parlay::sequence<vertex> frontier = {start};
  nested_seq frontiers;
  int num_threads = omp_get_max_threads();
  // std::cout << "Number of openmp threads: " << num_threads << std::endl;

  while (!frontier.empty()) {
    frontiers.push_back(frontier);

    // Estimate total output size
    size_t est_size = parlay::reduce(parlay::map(frontier, [&](vertex u) {
      return G[u].size();
    }));

    // Preallocate output buffer
    parlay::sequence<vertex> next;
    next.reserve(est_size);

    // Thread-local buffers to avoid contention
    std::vector<std::vector<vertex>> local_buffers(num_threads);

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      std::vector<vertex>& local = local_buffers[tid];

#pragma omp for schedule(dynamic, 64)
      for (size_t i = 0; i < frontier.size(); ++i) {
        vertex u = frontier[i];
        for (vertex v : G[u]) {
          bool expected = false;
          if (!visited[v] && visited[v].compare_exchange_strong(expected, true)) {
            local.push_back(v);
          }
        }
      }
    }

    // Flatten thread-local buffers into one sequence
    for (auto& buf : local_buffers) {
      next.insert(next.end(), buf.begin(), buf.end());
    }

    // Remove duplicates and continue
    // frontier = parlay::remove_duplicates(next);
    frontier = std::move(next);
  }

  return frontiers;
}
