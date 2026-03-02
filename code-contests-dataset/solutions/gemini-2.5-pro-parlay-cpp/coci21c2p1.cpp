#include <iostream>
#include <string>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/monoid.h>

// A struct to represent a single Kaučuk command.
// Using a struct improves readability and organization.
struct Command {
    int type; // 0 for section, 1 for subsection, 2 for subsubsection
    std::string title;
};

int main() {
    // Use fast I/O for performance in competitive programming.
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int n;
    std::cin >> n;

    // Step 1: Read all commands into a parlay::sequence.
    // This part is sequential as input must be read in order.
    parlay::sequence<Command> commands(n);
    for (int i = 0; i < n; ++i) {
        std::string type_str, title_str;
        std::cin >> type_str >> title_str;
        int type_val;
        if (type_str == "section") {
            type_val = 0;
        } else if (type_str == "subsection") {
            type_val = 1;
        } else { // "subsubsection"
            type_val = 2;
        }
        commands[i] = {type_val, title_str};
    }

    // The problem has sequential dependencies (numbering depends on previous lines).
    // We can break these dependencies and solve in parallel using prefix sums (scans).
    // The general idea is to precompute cumulative counts and parent locations for all lines.

    // Step 2: Create flag sequences (1 for true, 0 for false) for each command type.
    auto is_section = parlay::map(commands, [](const Command& cmd) { return cmd.type == 0 ? 1 : 0; });
    auto is_subsection = parlay::map(commands, [](const Command& cmd) { return cmd.type == 1 ? 1 : 0; });
    auto is_subsubsection = parlay::map(commands, [](const Command& cmd) { return cmd.type == 2 ? 1 : 0; });

    // Step 3: Calculate cumulative counts for each type using parallel inclusive scan.
    // `section_counts[i]` will be the total count of sections up to and including line i.
    // This directly gives the section number for any line.
    auto section_counts = parlay::scan_inclusive(is_section);
    // `subsection_total_counts[i]` will be the total count of subsections up to line i.
    auto subsection_total_counts = parlay::scan_inclusive(is_subsection);
    // `subsubsection_total_counts[i]` will be the total count of subsubsections up to line i.
    auto subsubsection_total_counts = parlay::scan_inclusive(is_subsubsection);

    // Step 4: Find the index of the last 'section' and 'subsection' for each line.
    // This is crucial for resetting the numbering within a new scope.
    // We create a sequence of indices, then use a scan with a max operator to propagate the last seen index.
    auto section_indices = parlay::tabulate(n, [&](long i) {
        return commands[i].type == 0 ? i : -1L;
    });
    auto last_section_indices = parlay::scan_inclusive(section_indices, parlay::maxm<long>());

    auto subsection_indices = parlay::tabulate(n, [&](long i) {
        return commands[i].type == 1 ? i : -1L;
    });
    auto last_subsection_indices = parlay::scan_inclusive(subsection_indices, parlay::maxm<long>());

    // Step 5: Generate the final numbered title strings in parallel using the precomputed tables.
    auto output_strings = parlay::tabulate(n, [&](size_t i) {
        const auto& command = commands[i];
        // The section number is the total count of sections up to this point.
        long s_num = section_counts[i];

        if (command.type == 0) { // It's a section
            return std::to_string(s_num) + " " + command.title;
        } else if (command.type == 1) { // It's a subsection
            long parent_section_idx = last_section_indices[i];
            // To get the local subsection number, we subtract the total number of subsections
            // that occurred before its parent section started.
            long count_before_parent = (parent_section_idx > 0) ? subsection_total_counts[parent_section_idx - 1] : 0;
            long ss_num = subsection_total_counts[i] - count_before_parent;
            return std::to_string(s_num) + "." + std::to_string(ss_num) + " " + command.title;
        } else { // It's a subsubsection
            long parent_subsection_idx = last_subsection_indices[i];
            
            // First, find the number of the containing subsection. This requires finding its
            // parent section and calculating its local number.
            long parent_section_for_ss_idx = last_section_indices[parent_subsection_idx];
            long count_before_s_for_ss = (parent_section_for_ss_idx > 0) ? subsection_total_counts[parent_section_for_ss_idx - 1] : 0;
            long ss_num = subsection_total_counts[parent_subsection_idx] - count_before_s_for_ss;

            // Second, find the number of the current subsubsection within its parent subsection.
            long count_before_ss = (parent_subsection_idx > 0) ? subsubsection_total_counts[parent_subsection_idx - 1] : 0;
            long sss_num = subsubsection_total_counts[i] - count_before_ss;

            return std::to_string(s_num) + "." + std::to_string(ss_num) + "." + std::to_string(sss_num) + " " + command.title;
        }
    });

    // Step 6: Print the results. This part is sequential as std::cout is not thread-safe
    // without manual synchronization.
    for (const auto& s : output_strings) {
        std::cout << s << "\n";
    }

    return 0;
}