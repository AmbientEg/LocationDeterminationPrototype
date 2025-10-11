import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon
import numpy as np
import os

# ===============================
# Load the processed data
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Data_PATH = os.path.join(BASE_DIR, "data", "processed", "trilateration_kalman_results.csv")
df = pd.read_csv(Data_PATH)

# Extract positions
x_smooth = df['X_smooth'].values
y_smooth = df['Y_smooth'].values
locations = df['Location'].values

# Shift so first smoothed point is at origin
origin_x = x_smooth[0]
origin_y = y_smooth[0]
x_smooth_shifted = x_smooth - origin_x
y_smooth_shifted = y_smooth - origin_y

# ===============================
# Floor plan and beacons
# ===============================
# Beacon positions based on floor plan (Column, Row)
beacon_positions = {
    "b3001": (5, 8), "b3002": (10, 4), "b3003": (14, 4),
    "b3004": (18, 4), "b3005": (10, 7), "b3006": (14, 7),
    "b3007": (18, 7), "b3008": (10, 10), "b3009": (4, 15),
    "b3010": (10, 15), "b3011": (14, 15), "b3012": (18, 15),
    "b3013": (23, 15)
}

# Scale the smoothed positions to match floor plan (25x17 units)
x_min, x_max = x_smooth_shifted.min(), x_smooth_shifted.max()
y_min, y_max = y_smooth_shifted.min(), y_smooth_shifted.max()

x_scaled = (x_smooth_shifted - x_min) / (x_max - x_min) * 25
y_scaled = (y_smooth_shifted - y_min) / (y_max - y_min) * 17

# ===============================
# Plotting
# ===============================
fig, ax = plt.subplots(figsize=(16, 12))
floor_color = '#f5f5f5'
wall_color = '#333333'
ax.set_facecolor(floor_color)

# Example room layout (adjust as needed)
stairwell = Rectangle((0, 0), 6, 8, linewidth=2, edgecolor=wall_color,
                      facecolor='#e0e0e0', alpha=0.3, label='Stairwell')
ax.add_patch(stairwell)

central = Rectangle((6, 0), 18, 15, linewidth=2, edgecolor=wall_color,
                    facecolor='white', alpha=0.2)
ax.add_patch(central)

right_room = Polygon([(22, 9), (25, 9), (25, 15), (22, 15)],
                     linewidth=2, edgecolor=wall_color, facecolor='#d4a574',
                     alpha=0.3, label='Room')
ax.add_patch(right_room)

# Plot beacons
for name, (bx, by) in beacon_positions.items():
    ax.scatter(bx, by, color='lime', s=300, marker='o',
               edgecolors='darkgreen', linewidth=2, zorder=5, label='Beacon' if name=='b3001' else '')
    ax.annotate(name, (bx, by), xytext=(5, -10), textcoords='offset points',
                fontsize=8, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

# Plot smoothed path
ax.plot(x_scaled, y_scaled, color='darkblue', linewidth=2.5, label='Smoothed path', zorder=3)
ax.scatter(x_scaled, y_scaled, color='darkblue', s=100, zorder=4, edgecolors='black', linewidth=0.5)

# Label each point
for i, loc in enumerate(locations):
    ax.annotate(loc, (x_scaled[i], y_scaled[i]), xytext=(5, 5), textcoords='offset points',
                fontsize=7, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Mark the origin
ax.scatter(0, 0, color='red', s=300, marker='*', label='Origin (start)', zorder=5, edgecolors='darkred', linewidth=1.5)

# Labels, grid, and formatting
ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
ax.set_title('Indoor Position Tracking - Overlay on Floor Map', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax.set_aspect('equal')

# Set axis limits
ax.set_xlim(0, 25)  # Columns A→Y
ax.set_ylim(17, 0)  # Rows 1→17 (flipped to match floor plan)

plt.tight_layout()
plt.show()
