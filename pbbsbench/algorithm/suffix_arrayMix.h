#include <math.h>
#include <omp.h> // For OpenMP support
#include "../parlay/parallel.h"
#include "../parlay/primitives.h"
#include "../parlay/internal/get_time.h"

constexpr bool verbose = false;

using uchar = unsigned char;
using uint128 = unsigned __int128;

template <typename indexT>
using ipair = std::pair<indexT,indexT>;

template <typename indexT>
struct seg {
  indexT start;
  indexT length;
  seg<indexT>(indexT s, indexT l) : start(s), length(l) {}
  seg<indexT>() {}
};

// Hybrid OpenMP + ParlayLib implementation of split_segment
template <typename indexT>
void split_segment(parlay::slice<seg<indexT>*,seg<indexT>*> segOut,
		   indexT start,
		   parlay::sequence<indexT> &ranks,
		   parlay::slice<ipair<indexT>*,ipair<indexT>*> Cs) {
  indexT l = segOut.size();
  if (l < 5000) { // Sequential version for small segments remains efficient
    indexT name = 0;
    ranks[Cs[0].second] = name + start + 1;
    for (indexT i=1; i < l; i++) {
      if (Cs[i-1].first != Cs[i].first) name = i;
      ranks[Cs[i].second] = name + start + 1;
    }

    name = 0;
    for (indexT i=1; i < l; i++) {
      if (Cs[i-1].first != Cs[i].first) {
	    segOut[i-1] = seg<indexT>(name+start,i-name);
	    name = i;
      } else segOut[i-1] = seg<indexT>(0,0);
    }
    segOut[l-1] = seg<indexT>(name+start,l-name);

  } else { // Hybrid parallel version
    auto names = parlay::sequence<indexT>::uninitialized(l);

    // Use OpenMP for this simple, data-parallel loop
    #pragma omp parallel for
    for (size_t i=1; i < l; i++) {
	    names[i] = (Cs[i].first != Cs[i-1].first) ? i : 0;
    }
    names[0] = 0;

    // ParlayLib's scan is a highly optimized primitive, best to keep it
    parlay::scan_inclusive_inplace(names, parlay::maxm<indexT>());

    // Use OpenMP to write new ranks
    #pragma omp parallel for
    for (size_t i=0; i < l; i++) {
	    ranks[Cs[i].second] = names[i]+start+1;
    }

    // Use OpenMP to calculate starts and lengths of new segments
    #pragma omp parallel for
    for (size_t i=1; i < l; i++) {
	    if (names[i] == i)
	      segOut[i-1] = seg<indexT>(start+names[i-1],i-names[i-1]);
	    else segOut[i-1] = seg<indexT>(0,0);
    }
    segOut[l-1] = seg<indexT>(start+names[l-1],l-names[l-1]);
  }
}

// Hybrid OpenMP + ParlayLib implementation of split_segment_top
template <class indexT>
parlay::sequence<ipair<indexT>>
split_segment_top(parlay::sequence<seg<indexT>> &segOut,
		  parlay::sequence<indexT> &ranks,
		  parlay::sequence<uint128> const &Cs) {
  size_t n = segOut.size();
  auto names = parlay::sequence<indexT>::uninitialized(n);
  size_t mask = ((((size_t) 1) << 32) - 1);

  // Use OpenMP to mark start of each segment
  #pragma omp parallel for
  for (size_t i=1; i < n; i++) {
      names[i] = ((Cs[i] >> 32) != (Cs[i-1] >> 32)) ? i : 0;
  }
  names[0] = 0;

  // Keep ParlayLib's optimized scan
  parlay::scan_inclusive_inplace(names, parlay::maxm<indexT>());

  auto C = parlay::sequence<ipair<indexT>>::uninitialized(n);
  // Use OpenMP to write new ranks and populate the pair sequence
  #pragma omp parallel for
  for (size_t i=0; i < n; i++) {
      ranks[Cs[i] & mask] = names[i]+1;
      C[i].second = Cs[i] & mask;
  }

  // Use OpenMP to get starts and lengths of new segments
  #pragma omp parallel for
  for (size_t i=1; i < n; i++) {
      if (names[i] == i)
	    segOut[i-1] = seg<indexT>(names[i-1],i-names[i-1]);
      else segOut[i-1] = seg<indexT>(0,0);
  }
  segOut[n-1] = seg<indexT>(names[n-1],n-names[n-1]);

  return C;
}

// Main hybrid suffix array function
template <class indexT, class UCharRange>
parlay::sequence<indexT> suffix_array(UCharRange const &ss) {
  parlay::internal::timer sa_timer("Suffix Array (Hybrid OMP+Parlay)", false);
  size_t n = ss.size();

  // renumber characters densely
  size_t pad = 48;
  parlay::sequence<indexT> flags(256, (indexT) 0);
  // Use OpenMP for this loop. Note: This involves a benign race condition,
  // as multiple threads might write 1 to the same location, which is acceptable.
  #pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
      if (!flags[ss[i]]) flags[ss[i]] = 1;
  }
  
  // Keep ParlayLib's scan for renumbering
  auto add = [&] (indexT a, indexT b) {return a + b;};
  indexT m;
  std::tie(flags, m) = parlay::scan(flags, parlay::make_monoid(add,(indexT) 1));

  // Use OpenMP to parallelize the creation of the padded string
  auto s = parlay::sequence<uchar>::uninitialized(n + pad);
  #pragma omp parallel for
  for (size_t i = 0; i < n + pad; i++) {
      s[i] = (i < n) ? flags[ss[i]] : 0;
  }

  if (verbose) std::cout << "distinct characters = " << m-1 << std::endl;

  // pack characters into 128-bit words
  double logm = log2((double) m);
  indexT nchars = floor(96.0/logm);

  // Use OpenMP to parallelize the initial packing
  auto Cl = parlay::sequence<uint128>::uninitialized(n);
  #pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
      uint128 r = s[i];
      for (indexT j=1; j < nchars; j++) r = r*m + s[i+j];
      Cl[i] = (r << 32) + i;
  }
  sa_timer.next("copy into 128bit int");

  // ParlayLib's sort is highly optimized and should be kept
  parlay::sort_inplace(Cl, std::less<uint128>());
  sa_timer.next("sort");

  // identify segments of equal values using the hybrid helper
  auto ranks = parlay::sequence<indexT>::uninitialized(n);
  auto seg_outs = parlay::sequence<seg<indexT>>::uninitialized(n); 
  parlay::sequence<ipair<indexT>> C = split_segment_top(seg_outs, ranks, Cl);
  Cl.clear();
  sa_timer.next("split top");

  indexT offset = nchars;
  uint round =0;
  indexT nKeys = n;

  while (1) {
    if (round++ > 40)
      throw std::runtime_error("Suffix Array: internal error, too many rounds");

    auto is_seg = [&] (seg<indexT> s) {return s.length > 1;};
    // Keep ParlayLib's filter
    parlay::sequence<seg<indexT>> Segs = parlay::filter(seg_outs.cut(0,nKeys), is_seg);
    indexT nSegs = Segs.size();
    if (nSegs == 0) break;
    sa_timer.next("filter and scan");

    auto offsets = parlay::sequence<indexT>::uninitialized(nSegs);
    
    // Use ParlayLib for the coarse-grained loop over segments
    parlay::parallel_for (0, nSegs, [&] (size_t i) {
	    indexT start = Segs[i].start;
	    indexT l = Segs[i].length;
	    auto Ci = C.cut(start, start + l);
	    offsets[i] = l;

	    // Within the Parlay task, use OpenMP for the fine-grained inner loop.
        // This creates nested parallelism.
        #pragma omp parallel for
	    for (size_t j = 0; j < l; j++) {
	        indexT o = Ci[j].second + offset;
	        Ci[j].first = (o >= n) ? 0 : ranks[o];
	    }

	    // Keep ParlayLib's sort for sorting within segments
	    auto less = [&] (ipair<indexT> A, ipair<indexT> B) {
	      return A.first < B.first;};
	    parlay::sort_inplace(Ci, less);
      });
    sa_timer.next("sort");

    // Keep ParlayLib's scan
    nKeys = parlay::scan_inplace(offsets, parlay::addm<indexT>());

    // Use ParlayLib to dispatch the split_segment tasks
    parlay::parallel_for (0, nSegs, [&] (size_t i) {
	    indexT start = Segs[i].start;
	    indexT l = Segs[i].length;
	    indexT o = offsets[i];
	    split_segment(seg_outs.cut(o, o + l),
		      start,
		      ranks,
		      C.cut(start, start+l));
      }, 100);
    sa_timer.next("split");
    
    if (verbose)
      std::cout << "length: " << offset << " keys remaining: " << nKeys << std::endl;
    
    offset = 2 * offset;
  }
  
  // Use OpenMP for the final simple copy
  #pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
      ranks[i] = C[i].second;
  }
  return ranks;
}


// #include <math.h>
// #include <omp.h> // Include OpenMP header

// #include "../parlay/parallel.h"
// #include "../parlay/primitives.h"
// #include "../parlay/internal/get_time.h"

// constexpr bool verbose = false;

// using uchar = unsigned char;
// using uint128 = unsigned __int128;

// template <typename indexT>
// using ipair = std::pair<indexT,indexT>;

// template <typename indexT>
// struct seg {
//   indexT start;
//   indexT length;
//   seg<indexT>(indexT s, indexT l) : start(s), length(l) {}
//   seg<indexT>() {}
// };

// // Uses OpenMP for simple parallel loops, but retains ParlayLib for the scan.
// template <typename indexT>
// void split_segment(parlay::slice<seg<indexT>*,seg<indexT>*> segOut,
// 		   indexT start,
// 		   parlay::sequence<indexT> &ranks,
// 		   parlay::slice<ipair<indexT>*,ipair<indexT>*> Cs) {
//   indexT l = segOut.size();
//   if (l < 5000) { // sequential version remains the same
//     indexT name = 0;
//     ranks[Cs[0].second] = name + start + 1;
//     for (indexT i=1; i < l; i++) {
//       if (Cs[i-1].first != Cs[i].first) name = i;
//       ranks[Cs[i].second] = name + start + 1;
//     }

//     name = 0;
//     for (indexT i=1; i < l; i++) {
//       if (Cs[i-1].first != Cs[i].first) {
// 	segOut[i-1] = seg<indexT>(name+start,i-name);
// 	name = i;
//       } else segOut[i-1] = seg<indexT>(0,0);
//     }
//     segOut[l-1] = seg<indexT>(name+start,l-name);

//   } else { // parallel version using OpenMP for loops
//     auto names = parlay::sequence<indexT>::uninitialized(l);

//     // mark start of each segment with equal keys
//     #pragma omp parallel for
//     for (size_t i=1; i < l; i++) {
// 	names[i] = (Cs[i].first != Cs[i-1].first) ? i : 0;
//     }
//     names[0] = 0;

//     // scan remains a highly optimized ParlayLib primitive
//     parlay::scan_inclusive_inplace(names, parlay::maxm<indexT>());

//     // write new rank into original location
//     #pragma omp parallel for
//     for (size_t i=0; i < l; i++) {
// 	ranks[Cs[i].second] = names[i]+start+1;
//     }

//     // get starts and lengths of new segments
//     #pragma omp parallel for
//     for (size_t i=1; i < l; i++) {
// 	if (names[i] == i)
// 	  segOut[i-1] = seg<indexT>(start+names[i-1],i-names[i-1]);
// 	else segOut[i-1] = seg<indexT>(0,0);
//     }
//     segOut[l-1] = seg<indexT>(start+names[l-1],l-names[l-1]);
//   }
// }

// // Uses OpenMP for simple parallel loops, but retains ParlayLib for the scan.
// template <class indexT>
// parlay::sequence<ipair<indexT>>
// split_segment_top(parlay::sequence<seg<indexT>> &segOut,
// 		  parlay::sequence<indexT> &ranks,
// 		  parlay::sequence<uint128> const &Cs) {
//   size_t n = segOut.size();
//   auto names = parlay::sequence<indexT>::uninitialized(n);
//   size_t mask = ((((size_t) 1) << 32) - 1);

//   // mark start of each segment with equal keys
//   #pragma omp parallel for
//   for (size_t i=1; i < n; i++) {
//       names[i] = ((Cs[i] >> 32) != (Cs[i-1] >> 32)) ? i : 0;
//   }
//   names[0] = 0;

//   // scan remains a highly optimized ParlayLib primitive
//   parlay::scan_inclusive_inplace(names, parlay::maxm<indexT>());

//   auto C = parlay::sequence<ipair<indexT>>::uninitialized(n);
//   // write new rank into original location
//   #pragma omp parallel for
//   for (size_t i=0; i < n; i++) {
//       ranks[Cs[i] & mask] = names[i]+1;
//       C[i].second = Cs[i] & mask;
//   }

//   // get starts and lengths of new segments
//   #pragma omp parallel for
//   for (size_t i=1; i < n; i++) {
//       if (names[i] == i)
// 	segOut[i-1] = seg<indexT>(names[i-1],i-names[i-1]);
//       else segOut[i-1] = seg<indexT>(0,0);
//   }
//   segOut[n-1] = seg<indexT>(names[n-1],n-names[n-1]);

//   return C;
// }

// // Main function updated to use a hybrid OpenMP + ParlayLib approach.
// template <class indexT, class UCharRange>
// parlay::sequence<indexT> suffix_array(UCharRange const &ss) {
//   parlay::internal::timer sa_timer("Suffix Array", false);
//   size_t n = ss.size();

//   // renumber characters densely, using OpenMP with an atomic write for correctness
//   // start numbering at 1 leaving 0 to indicate end-of-string
//   size_t pad = 48;
//   parlay::sequence<indexT> flags(256, (indexT) 0);
//   #pragma omp parallel for
//   for (size_t i = 0; i < n; i++) {
//       // Use an atomic write to prevent data races. The original implementation
//       // had a race condition here, although it was often harmless in practice.
//       #pragma omp atomic write
//       flags[ss[i]] = 1;
//   }
//   auto add = [&] (indexT a, indexT b) {return a + b;};
//   indexT m;
//   std::tie(flags, m) = parlay::scan(flags, parlay::make_monoid(add,(indexT) 1));

//   // pad the end of string with 0s
//   auto s = parlay::tabulate(n + pad, [&] (size_t i) -> uchar {
//       return (i < n) ? flags[ss[i]] : 0;});

//   if (verbose) std::cout << "distinct characters = " << m-1 << std::endl;

//   // pack characters into 128-bit word, along with the location i
//   // 96 bits for characters, and 32 for location
//   double logm = log2((double) m);
//   indexT nchars = floor(96.0/logm);

//   // Initialize with OpenMP instead of parlay::tabulate
//   auto Cl = parlay::sequence<uint128>::uninitialized(n);
//   #pragma omp parallel for
//   for (size_t i = 0; i < n; i++) {
//       uint128 r = s[i];
//       for (indexT j=1; j < nchars; j++) r = r*m + s[i+j];
//       Cl[i] = (r << 32) + i;
//   }
//   sa_timer.next("copy into 128bit int");

//   // sort based on packed words - ParlayLib sort is highly optimized
//   parlay::sort_inplace(Cl, std::less<uint128>());
//   sa_timer.next("sort");

//   // identify segments of equal values
//   auto ranks = parlay::sequence<indexT>::uninitialized(n);
//   auto seg_outs = parlay::sequence<seg<indexT>>::uninitialized(n); 
//   parlay::sequence<ipair<indexT>> C = split_segment_top(seg_outs, ranks, Cl);
//   Cl.clear();
//   sa_timer.next("split top");

//   indexT offset = nchars;
//   uint round =0;
//   indexT nKeys = n;

//   // offset is how many characters for each suffix have already been sorted
//   // each round doubles offset so there should be at most log n rounds
//   // The segments keep regions that have not yet been fully sorted
//   while (1) {
//     if (round++ > 40)
//       throw std::runtime_error("Suffix Array: internal error, too many rounds");

//     auto is_seg = [&] (seg<indexT> s) {return s.length > 1;};
//     // only keep segments that are longer than 1 (otherwise already sorted)
//     // ParlayLib filter is efficient for this.
//     parlay::sequence<seg<indexT>> Segs = parlay::filter(seg_outs.cut(0,nKeys), is_seg);
//     indexT nSegs = Segs.size();
//     if (nSegs == 0) break;
//     sa_timer.next("filter and scan");

//     auto offsets = parlay::sequence<indexT>::uninitialized(nSegs);
    
//     // Use OpenMP for the coarse-grained loop over segments.
//     // Dynamic schedule is used for better load balancing since segments can vary in size.
//     #pragma omp parallel for schedule(dynamic, 1)
//     for (size_t i = 0; i < nSegs; i++) {
//         indexT start = Segs[i].start;
//         indexT l = Segs[i].length;
//         auto Ci = C.cut(start, start + l);
//         offsets[i] = l;

//         // This inner loop remains a ParlayLib call, creating nested parallelism.
//         parlay::parallel_for (0, l, [&] (size_t j) {
//             indexT o = Ci[j].second + offset;
//             Ci[j].first = (o >= n) ? 0 : ranks[o];
//           }, 100);

//         // Sort within each segment also uses ParlayLib's optimized sort.
//         auto less = [&] (ipair<indexT> A, ipair<indexT> B) {
//           return A.first < B.first;};
//         parlay::sort_inplace(Ci, less);
//     }
//     sa_timer.next("sort");

//     // starting offset for each segment - ParlayLib scan is ideal.
//     nKeys = parlay::scan_inplace(offsets, parlay::addm<indexT>());

//     // Split each segment into subsegments if neighbors differ.
//     // Again, use OpenMP for the outer loop over segments.
//     #pragma omp parallel for schedule(dynamic, 1)
//     for (size_t i = 0; i < nSegs; i++) {
//         indexT start = Segs[i].start;
//         indexT l = Segs[i].length;
//         indexT o = offsets[i];
//         split_segment(seg_outs.cut(o, o + l),
//               start,
//               ranks,
//               C.cut(start, start+l));
//     }
//     sa_timer.next("split");
    
//     if (verbose)
//       std::cout << "length: " << offset << " keys remaining: " << nKeys << std::endl;
    
//     offset = 2 * offset;
//   }

//   // Final copy of results using OpenMP.
//   #pragma omp parallel for
//   for (size_t i = 0; i < n; i++) {
//       ranks[i] = C[i].second;
//   }
//   return ranks;
// }
