#include <iostream>
#include <string>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/internal/get_time.h>

#include "bfs_mixed.h"
#include "../helper/graph_utils.h"

// **************************************************************
// Driver
// **************************************************************
using vertex = int;
using nested_seq = parlay::sequence<parlay::sequence<vertex>>;
using graph = nested_seq;
using utils = graph_utils<vertex>;

int checkBFS_nested_seq(vertex start, const nested_seq& G, const nested_seq& levels) {
  size_t n = G.size();
  size_t max_level = levels.size();

  // Build vertex_level map
  parlay::sequence<int> vertex_level(n, -1);
  for (size_t lvl = 0; lvl < max_level; ++lvl) {
    for (vertex v : levels[lvl]) {
      vertex_level[v] = (int)lvl;
    }
  }

  if (vertex_level[start] != 0) {
    std::cerr << "BFSCheck: Start vertex not at level 0\n";
    return 1;
  }

  // For each level > 0
  for (size_t lvl = 1; lvl < max_level; ++lvl) {
    // Build hash set of all neighbors of level (lvl-1) vertices
    auto prev_level_vertices = levels[lvl - 1];
    size_t total_neighbors = 0;
    for (vertex u : prev_level_vertices) total_neighbors += G[u].size();

    parlay::sequence<vertex> neighbors(total_neighbors);
    size_t pos = 0;
    for (vertex u : prev_level_vertices) {
      for (vertex w : G[u]) {
        neighbors[pos++] = w;
      }
    }

    // Build a hash set for fast membership test
    auto neighbor_set = parlay::sort(parlay::unique(parlay::sort(neighbors)));

    // For each vertex v in current level, check if v is in neighbor_set
    for (vertex v : levels[lvl]) {
      if (!std::binary_search(neighbor_set.begin(), neighbor_set.end(), v)) {
        std::cerr << "BFSCheck: Vertex " << v << " at level " << lvl
                  << " has no parent in level " << (lvl - 1) << std::endl;
        return 1;
      }
    }
  }

  std::cout << "BFSCheck: passed!" << std::endl;
  return 0;
}


int main(int argc, char* argv[]) {
  auto usage = "Usage: BFS <n> || BFS <filename>";
  if (argc != 2) std::cout << usage << std::endl;
  else {
    long n = 0;
    graph G;
    try { n = std::stol(argv[1]); }
    catch (...) {}
    if (n == 0) {
      G = utils::read_symmetric_graph_from_file(argv[1]);
      n = G.size();
    } else {
      G = utils::rmat_graph(n, 20*n);
    }
    utils::print_graph_stats(G);
    nested_seq result;
    parlay::internal::timer t("Time");
    for (int i=0; i < 3; i++) {
      result = BFS(1, G);
      t.next("BFS");
    

      long visited = parlay::reduce(parlay::map(result, parlay::size_of()));
      std::cout << "num vertices visited: " << visited << std::endl;

    // std::cout << "adjacency list" << std::endl;
    // for (int i = 0; i < G.size(); i++) {
    //   std::cout << i << ": ";
    //   for (const auto& v : G[i]) {
    //     std::cout << v << " ";
    //   }
    //   std::cout << std::endl;
    // }

    // for (int i = 0; i < result.size(); i++) {
    //   std::cout << "Level " << i << ": ";
    //   for (const auto& v : result[i]) {
    //     std::cout << v << " ";
    //   }
    //   std::cout << std::endl;
    // }

      // Call the correctness checker
      int check_result = checkBFS_nested_seq(1, G, result);
      if (check_result != 0) {
        std::cerr << "BFS validation failed!" << std::endl;
        return 1;
      }
    }
  }
}

