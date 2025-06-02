import random
import csv

# Define the three target regions by their bounding boxes
regions = [
    {'x_min': 0, 'x_max': 2, 'y_min': 0, 'y_max': 2},  # Region 1: (0,0)-(2,2)
    {'x_min': 0, 'x_max': 2, 'y_min': 3, 'y_max': 5},  # Region 2: (0,3)-(2,5)
    {'x_min': 5, 'x_max': 6, 'y_min': 0, 'y_max': 2},  # Region 3: (5,0)-(6,2)
    {'x_min': 5, 'x_max': 6, 'y_min': 3, 'y_max': 5},  # Region 3: (5,3)-(6,5)
    {'x_min': 3, 'x_max': 4, 'y_min': 2, 'y_max': 3},  # Region 3: (3,2)-(4,3)
]

# Number of points to generate
n_points = 500

# Generate random points across the three regions
data = []
for _ in range(n_points):
    region = random.choice(regions)
    x = random.uniform(region['x_min'], region['x_max'])
    y = random.uniform(region['y_min'], region['y_max'])
    data.append((x, y))

# Write to CSV file
output_file = 'random_coordinates.csv'
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['X_coordinate', 'y_coordinate'])
    writer.writerows(data)

print(f"Dataset of {n_points} points saved to '{output_file}'")

