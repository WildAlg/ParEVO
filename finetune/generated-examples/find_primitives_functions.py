import re

with open("../../parlaylib/include/parlay/primitives.h", "r") as f:
    contents = f.read()

# Updated pattern:
pattern = re.compile(
    r'template\s*<[^>]+>\s*'     # Match template declaration
    r'(?:.|\n)*?'                # Non-greedy match across lines
    r'auto\s+(\w+)\s*\(',        # Capture function name
    re.MULTILINE
)

matches = pattern.findall(contents)

for match in matches:
    print(match)
