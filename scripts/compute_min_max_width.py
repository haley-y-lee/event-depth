filename = "../../DENSE/small_upsampled/seq/frames/boundaries.txt"

min_diff = float('inf')
max_diff = float('-inf')

with open(filename, 'r') as f:
    for line in f:
        if not line.strip():
            continue  # skip empty lines
        lower, upper = map(int, line.strip().split())
        diff = abs(upper - lower)

        min_diff = min(min_diff, diff)
        max_diff = max(max_diff, diff)

print(f"Minimum difference: {min_diff}")
print(f"Maximum difference: {max_diff}")