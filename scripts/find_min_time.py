filename = "../../DENSE/train_upsampled_2/train_sequence_01_town02/timestamps.txt"

with open(filename, "r") as f:
    numbers = [float(line.strip()) for line in f if line.strip()]

# Sort the numbers (if needed — remove if already sorted in file)
numbers.sort()

# Compute differences between consecutive numbers
diffs = [b - a for a, b in zip(numbers[:-1], numbers[1:])]

# Find minimum difference
min_diff = min(diffs)
average_diff = sum(diffs) / len(diffs)
# breakpoint()

print("Minimum difference:", min_diff)
print("Average difference:", average_diff)