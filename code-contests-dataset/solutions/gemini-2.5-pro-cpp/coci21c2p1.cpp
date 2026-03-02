#include <iostream>
#include <string>
#include <vector>

// Core parlay headers for parallel algorithms and data structures
#include <parlay/primitives.h>
#include <parlay/sequence.h>

// Function to convert command string to an integer type for efficient processing.
// section -> 0, subsection -> 1, subsubsection -> 2
int command_to_type(const std::string& cmd) {
    if (cmd == "section") return 0;
    if (cmd == "subsection") return 1;
    if (cmd == "subsubsection") return 2;
    return -1; // Should not happen based on problem constraints
}

int main() {
    // Use fast I/O
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int n;
    std::cin >> n;
    
    if (n == 0) {
        return 0;
    }

    // Read all input commands and titles into a sequence of pairs.
    // This part is sequential but necessary for input processing.
    parlay::sequence<std::pair<std::string, std::string>> commands(n);
    for (int i = 0; i < n; ++i) {
        std::cin >> commands[i].first >> commands[i].second;
    }

    // Convert command strings to integer types in parallel.
    auto types = parlay::map(commands, [](const auto& p) {
        return command_to_type(p.first);
    });

    // --- Core parallel computation using Parlay ---

    // 1. Calculate section numbers (X) for all commands.
    // Create a sequence of flags: 1 if the command is a 'section', 0 otherwise.
    auto is_section = parlay::map(types, [](int type) { return (type == 0) ? 1 : 0; });
    // An inclusive scan (prefix sum) on this sequence gives the section number for each command.
    auto section_numbers = parlay::scan_inclusive(is_section);

    // 2. Calculate subsection numbers (Y) for all commands.
    parlay::sequence<int> subsection_numbers(n);
    {
        // Group commands by their section number to process each section independently.
        auto groups = parlay::group_by_key(parlay::zip(section_numbers, parlay::iota<size_t>(n)));
        
        // Process each group (section) in parallel.
        parlay::parallel_for(0, groups.size(), [&](size_t i) {
            auto const& group_indices = groups[i].second;
            
            // Within a section, identify subsections.
            auto is_subsection_in_group = parlay::map(group_indices, [&](size_t original_idx) {
                return (types[original_idx] == 1) ? 1 : 0;
            });
            
            // Calculate local subsection counts within this section.
            auto local_subsection_counts = parlay::scan_inclusive(is_subsection_in_group);
            
            // Propagate the last subsection number forward. This is needed for subsubsections
            // which belong to a subsection but don't increment the counter themselves.
            // A max-scan achieves this propagation.
            auto propagated_counts = parlay::scan_inclusive(local_subsection_counts, parlay::maximum<int>());
            
            // Write the calculated subsection numbers back to the global sequence.
            parlay::parallel_for(0, group_indices.size(), [&](size_t j) {
                subsection_numbers[group_indices[j]] = propagated_counts[j];
            });
        });
    }

    // 3. Calculate subsubsection numbers (Z) for all commands.
    parlay::sequence<int> subsubsection_numbers(n);
    {
        // Segments for subsubsection counting reset at every 'section' or 'subsection'.
        auto is_z_segment_start = parlay::map(types, [](int type) { return (type == 0 || type == 1) ? 1 : 0; });
        auto z_segment_ids = parlay::scan_inclusive(is_z_segment_start);

        // Group commands by their segment ID.
        auto groups = parlay::group_by_key(parlay::zip(z_segment_ids, parlay::iota<size_t>(n)));

        // Process each group in parallel.
        parlay::parallel_for(0, groups.size(), [&](size_t i) {
            auto const& group_indices = groups[i].second;

            // Within a group, identify subsubsections.
            auto is_subsubsection_in_group = parlay::map(group_indices, [&](size_t original_idx) {
                return (types[original_idx] == 2) ? 1 : 0;
            });

            // Calculate local subsubsection counts.
            auto local_subsubsection_counts = parlay::scan_inclusive(is_subsubsection_in_group);
            
            // Write results back. No propagation is needed as no lower-level headings depend on this number.
            parlay::parallel_for(0, group_indices.size(), [&](size_t j) {
                subsubsection_numbers[group_indices[j]] = local_subsubsection_counts[j];
            });
        });
    }

    // 4. Generate all output strings in parallel.
    auto output_lines = parlay::tabulate(n, [&](size_t i) {
        std::string line;
        int type = types[i];
        const auto& title = commands[i].second;

        if (type == 0) { // section
            line = std::to_string(section_numbers[i]) + " " + title;
        } else if (type == 1) { // subsection
            line = std::to_string(section_numbers[i]) + "." + std::to_string(subsection_numbers[i]) + " " + title;
        } else { // subsubsection
            line = std::to_string(section_numbers[i]) + "." + std::to_string(subsection_numbers[i]) + "." + std::to_string(subsubsection_numbers[i]) + " " + title;
        }
        return line;
    });

    // 5. Print the results sequentially to ensure correct order and avoid race conditions on std::cout.
    for (int i = 0; i < n; ++i) {
        std::cout << output_lines[i] << "\n";
    }

    return 0;
}