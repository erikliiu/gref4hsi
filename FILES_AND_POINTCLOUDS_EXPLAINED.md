# Understanding the Files and Point Clouds

## Part 1: What Gets Shot by Rays?

### We shoot rays at: `model.ply` ✅

```
HSI Camera Position (ECEF coordinates)
         |
         | Rays cast downward
         ↓
    [model.ply MESH]  ← RAY INTERSECTION happens here!
    (3D triangulated surface)
         |
         | Intersection found at (X, Y, Z)
         ↓
    Intersection Point saved to H5 file
```

### `dummy_dem.tif` is NOT used for ray tracing!

It's just a raster version of the same seafloor for visualization.

---

## Part 2: The Two "Seafloor Representations"

### They're TWINS representing the same seafloor:

```
Original Data: Altimeter Point Cloud
              (4,682 measurements)
                      |
        ┌─────────────┴─────────────┐
        ↓                           ↓
   [3D Mesh]                   [2D Raster]
   model.ply                  dummy_dem.tif
        |                           |
   Used for:                   Used for:
   - Ray tracing               - Visualization
   - Finding intersections     - GIS software
   - Trimesh/PyVista          - QGIS display
                               - Plotting
```

### Example to clarify:

Imagine the seafloor is a crumpled piece of paper:

- **`model.ply`**: A 3D paper model made of triangular panels
  - You can trace a laser beam and see where it hits
  - Good for: "Where does this ray intersect the surface?"

- **`dummy_dem.tif`**: A photograph of that paper from above
  - Each pixel shows the height at that location
  - Good for: "Show me a map of elevations"

**Same paper, different representations!**

---

## Part 3: The ECEF Point Cloud from HSI

### YES, it's a point cloud! Here's what happens:

```
Step 1: Ray Tracing
━━━━━━━━━━━━━━━━━━
HSI has 1290 pixels per line, 500 lines
= 645,000 rays shot at model.ply

Each ray that hits creates an intersection point:
Ray 1 → Hit at (X₁, Y₁, Z₁) in ECEF
Ray 2 → Hit at (X₂, Y₂, Z₂) in ECEF
Ray 3 → Hit at (X₃, Y₃, Z₃) in ECEF
...
Ray 645,000 → Hit at (X₆₄₅₀₀₀, Y₆₄₅₀₀₀, Z₆₄₅₀₀₀) in ECEF


Step 2: Save to H5 file
━━━━━━━━━━━━━━━━━━━━━
All intersection points saved as a grid:
processed/georef/points_ecef_crs
Shape: (500 lines, 1290 pixels, 3)
       [  track,    slit,     XYZ]

This is a STRUCTURED point cloud:
- Each point = where a pixel "sees" the seafloor
- Grid structure = maintains image geometry
- 3D coordinates = real-world position in ECEF


Step 3: Visualization
━━━━━━━━━━━━━━━━━━━
Load points_ecef_crs from H5
Flatten to 1D array: 645,000 points
Convert ECEF → UTM
Plot as scatter: 645,000 dots showing where HSI saw seafloor
```

---

## Part 4: Two Point Clouds in Your System

### Point Cloud #1: Altimeter Measurements
- **Source**: Altimeter sensor on EELY
- **Points**: 4,682 measurements (from both H5 files)
- **Purpose**: Build the DEM/mesh of seafloor
- **Saved as**: 
  - `model.xyz` (raw points)
  - `model.ply` (triangulated mesh)
  - `dummy_dem.tif` (rasterized)
- **Represents**: The seafloor terrain itself

### Point Cloud #2: HSI Ray Intersections
- **Source**: Where HSI camera rays hit the seafloor
- **Points**: ~645,000 per H5 file (if 100% success)
- **Purpose**: Geolocate each HSI pixel
- **Saved as**: `processed/georef/points_ecef_crs` in H5 file
- **Represents**: Where each camera pixel "looks at" on the seafloor

---

## Part 5: The Complete Picture

```
┌─────────────────────────────────────────────────────────────┐
│ EELY Vehicle with HSI Camera and Altimeter                  │
└──────────────┬─────────────────────┬────────────────────────┘
               ↓                     ↓
        [HSI Camera]           [Altimeter]
               |                     |
               |                     ↓
               |           Measures seafloor
               |           distance (4,682 times)
               |                     |
               |                     ↓
               |           Build model.ply mesh
               |           (3D triangulated surface)
               |                     |
               ↓                     ↓
        Shoot 645,000 rays    ←━━━━━┘
        at model.ply mesh     (Ray tracing target)
               |
               ↓
        Find 645,000 intersections
        (where each pixel sees seafloor)
               |
               ↓
        Save to H5 file as
        points_ecef_crs
        (Point cloud #2)
```

---

## Part 6: Why Your Visualization Shows a Problem

```
Left Plot: dummy_dem.tif (raster view of Point Cloud #1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shows: Where altimeter measured seafloor
Location: North = 6,741,880 - 6,741,890 m


Right Plot: points_ecef_crs (Point Cloud #2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shows: Where HSI rays hit seafloor  
Location: North = 6,741,900 m

❌ PROBLEM: 10-20m spatial mismatch!
```

The HSI intersections are **north of the DEM**, which means:
1. The rays are hitting something, but not the actual seafloor terrain
2. There's a navigation/coordinate transformation error
3. OR the max_ray_length is too short and rays are hitting the mesh boundary/artifacts

---

## Summary Table

| File | Type | Used For | Point Cloud? |
|------|------|----------|--------------|
| `model.ply` | 3D mesh | ✅ Ray tracing target | Vertices form a point cloud |
| `dummy_dem.tif` | 2D raster | Visualization only | No (gridded elevations) |
| `model.xyz` | Text file | Point cloud export | Yes (raw XYZ coordinates) |
| `points_ecef_crs` | H5 dataset | HSI pixel geolocation | Yes (intersection points) |

**Key takeaway**: `model.ply` is the ray tracing target, and `points_ecef_crs` is the resulting point cloud of where rays hit.
