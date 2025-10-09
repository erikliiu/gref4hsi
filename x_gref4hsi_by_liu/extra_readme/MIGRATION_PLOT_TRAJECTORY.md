# Migration: plot_georef_with_trajectory() → utils/georef.py

## Date: 2025-10-08

## Change Summary

Moved `plot_georef_with_trajectory()` from standalone file to be a method of the `CombinedTransectCube` class.

### Before
```python
from dev.dev3_2D_georef_plot import plot_georef_with_trajectory

fig, ax = plot_georef_with_trajectory(
    cube,
    coordinate_system="LATLON",
    show_trajectory=True
)
```

### After
```python
# No import needed! Already part of the class

fig, ax = cube.plot_georef_with_trajectory(
    coordinate_system="LATLON",
    show_trajectory=True
)
```

## Why This Is Better

1. **Better Organization**: The function is now with all other plotting methods in `utils/georef.py`
2. **No Imports Needed**: It's a method of `CombinedTransectCube`, so you get it automatically when you load the transect
3. **Cleaner API**: Use `cube.plot_georef_with_trajectory()` instead of passing `cube` as an argument
4. **Consistency**: Matches the pattern of `cube.plot_georef()`, `cube.describe()`, etc.

## Files Modified

1. **`utils/georef.py`**: Added `plot_georef_with_trajectory()` method to `CombinedTransectCube` class (after line 715)
2. **`dev/dev1_plot_hsi.ipynb`**: Updated cell 5 to use the new method syntax (removed import, changed function call)

## Old File Status

**`dev/dev3_2D_georef_plot.py`**: Can now be deleted or kept as a backup. The function is fully integrated into the main codebase.

## Testing

To verify the change works:
```python
# Cell 2: Import georef
from utils.georef import *

# Cell 3: Load transect
transect = load_transect(r"path\to\output")
cube = transect.select_files([...])

# Cell 5: Plot with trajectory (NEW SYNTAX)
fig, ax = cube.plot_georef_with_trajectory(
    coordinate_system="LATLON",
    show_trajectory=True
)
```

## Benefits

✅ **Better code organization**: Related functionality grouped together  
✅ **Easier to use**: No separate import needed  
✅ **More maintainable**: Changes to plotting logic in one place  
✅ **Follows OOP principles**: Method belongs to the class it operates on
