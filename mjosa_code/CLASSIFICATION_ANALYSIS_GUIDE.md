# Classification Analysis Implementation Summary

## Overview
Added comprehensive analysis capabilities to all 4 classification notebooks to support narrative building with detailed metrics, visualizations, and statistics.

## What Was Added

### 1. New Utility Module
**File**: `mjosa_code/utils/uhi/classification_analysis.py`

**Function**: `analyze_classification_results(...)`

**Capabilities**:
- ✅ Per-class pixel counts and percentages
- ✅ Before/after filtering comparisons
- ✅ Unknown/unclassified pixel statistics
- ✅ Filtering impact analysis (pixels removed, % removed per class)
- ✅ Connected component cluster statistics (requires scipy):
  - Number of clusters per class
  - Largest/smallest/mean cluster size
  - Full cluster size distributions
- ✅ Visualizations:
  - Bar chart: before vs after filtering
  - Pie chart: final class distribution
  - Histogram: cluster size distribution
- ✅ CSV export of summary statistics
- ✅ Returns metrics dictionary for programmatic access

### 2. Added Analysis Cells to Notebooks

#### **Notebook 6: `6_svm_057_general.ipynb`**
- **Model**: 057 trained on 057 (3 classes: dark, sediment, bombs)
- **Purpose**: Baseline model performance
- **CSV Output**: `./saved_data/classification_summary_057_on_057.csv`
- **Location**: Bottom of notebook (2 new cells: markdown header + analysis)

#### **Notebook 9: `9_svm_028.ipynb`**
- **Model**: 028 trained on 028 (5 classes: sediment, rust, dark_bomb, dark_pit, halo)
- **Purpose**: Baseline model with more classes, crop-based analysis
- **Special Handling**: Analyzes each of 4 crop regions separately
- **CSV Outputs**: 
  - `./saved_data/classification_summary_028_crop_1.csv`
  - `./saved_data/classification_summary_028_crop_2.csv`
  - `./saved_data/classification_summary_028_crop_3.csv`
  - `./saved_data/classification_summary_028_crop_4.csv`
- **Location**: Bottom of notebook (2 new cells: markdown header + analysis)

#### **Notebook 10: `10_apply_028_model_to_057.ipynb`**
- **Model**: 028 model applied to 057 data (cross-transect transfer)
- **Purpose**: Test generalization of 5-class model to different site
- **CSV Output**: `./saved_data/classification_summary_028_to_057.csv`
- **Location**: Bottom of notebook (2 new cells: markdown header + analysis)

#### **Notebook 11: `11_apply_057_model_to_028.ipynb`**
- **Model**: 057 model applied to 028 crop regions (cross-transect transfer)
- **Purpose**: Test if simpler 3-class model captures features at different site
- **CSV Output**: `./saved_data/classification_summary_057_to_028_latest_crop.csv`
- **Special Note**: Analyzes most recent crop (run after each crop classification for multiple analyses)
- **Location**: Bottom of notebook (2 new cells: markdown header + analysis)

## How to Use

### Step 1: Run a Classification Notebook
Open any of the 4 notebooks and run all cells from top to bottom. The classification and filtering steps will populate the required variables.

### Step 2: Run the Analysis Cell
Scroll to the **very bottom** of the notebook and run the final cell. It will:
1. Import the analysis function
2. Locate classification results from the notebook
3. Generate comprehensive statistics (printed to console)
4. Display 2-3 plots (bar chart, pie chart, histogram if applicable)
5. Save a CSV summary to `./saved_data/`
6. Print a narrative summary with key insights

### Step 3: Use the Metrics for Your Narrative
The analysis provides:
- **Quantitative data**: exact pixel counts, percentages, cluster statistics
- **Visual evidence**: charts and plots for figures
- **CSV exports**: for tables and further analysis
- **Comparison points**: run all 4 notebooks to compare:
  - Baseline performance (6 and 9)
  - Cross-transfer generalization (10 and 11)
  - 3-class vs 5-class models
  - Same-site vs different-site application

## Example Output

When you run the analysis cell, you'll see:

```
================================================================================
CLASSIFICATION ANALYSIS SUMMARY
================================================================================
Total pixels in map: 1,234,567
Unknown / unclassified (before): 12,345 pixels (1.00%)
Unknown / unclassified (after): 8,901 pixels (0.72%)

Per-class counts (before filtering):
  sediment: 654,321 pixels (53.00%)
  rust: 234,567 pixels (19.00%)
  dark_bomb: 123,456 pixels (10.00%)
  ...

Per-class counts (after filtering):
  sediment: 654,000 pixels (52.97%)
  rust: 234,000 pixels (18.96%)
  dark_bomb: 100,000 pixels (8.10%)
  ...

Filtering impact (removed pixels):
  dark_bomb: removed 23,456 px (19.00%)
  dark_pit: removed 12,345 px (15.00%)
  ...

Cluster statistics (connected components):
  sediment: 45 clusters, largest=543,210 px, smallest=10 px, mean=14,533.3 px
  rust: 123 clusters, largest=123,456 px, smallest=5 px, mean=1,902.2 px
  ...

💾 Saved CSV summary to: ./saved_data/classification_summary_057_on_057.csv

[Bar chart displayed]
[Pie chart displayed]
[Histogram displayed]

================================================================================
📝 NARRATIVE SUMMARY
================================================================================
This is the baseline 057 model applied to its own training transect.
Total pixels classified: 1,234,567
  • sediment: 654,000 pixels (52.97%) after filtering
  • rust: 234,000 pixels (18.96%) after filtering
  ...
================================================================================
✅ Analysis complete. Returned metrics dictionary.
================================================================================
```

## Key Insights for Your Narrative

### Comparing All 4 Classifications:

1. **Baseline Performance (Notebooks 6 & 9)**:
   - How well do models perform on their own training transects?
   - What's the class distribution in each site?
   - How much filtering is needed?

2. **Cross-Transfer Generalization (Notebooks 10 & 11)**:
   - Does the 028 model (5 classes) work on 057 site?
   - Does the 057 model (3 classes) capture features at 028 site?
   - Which classes transfer well vs poorly?

3. **Model Complexity**:
   - 3-class (057) vs 5-class (028) models
   - Does the simpler model capture the essential features?
   - Are the additional classes in 028 necessary?

4. **Spatial Patterns** (from cluster statistics):
   - Large contiguous regions vs scattered pixels
   - Which classes form coherent patches?
   - Filtering effectiveness per class

## Troubleshooting

### Error: "classification_results not found"
**Solution**: Run all classification cells in the notebook before running the analysis cell.

### Error: "scipy.ndimage not available"
**Solution**: Cluster statistics will be skipped, but all other metrics and plots work fine. To enable:
```bash
pip install scipy
```

### Notebook 9: Want to analyze all crops?
**Solution**: The analysis cell is designed to analyze all 4 crops automatically if `multiclass_crop_results` exists. Just run the cell once after all crop classifications are complete.

### Notebook 11: Only seeing one crop?
**Solution**: The cell analyzes the most recent crop stored in `results`. To analyze multiple crops, save the results dict after each crop or re-run the analysis cell after each crop classification.

## Files Created/Modified

### New Files:
- `mjosa_code/utils/uhi/classification_analysis.py` (346 lines, standalone module)

### Modified Files:
- `mjosa_code/notebooks/6_svm_057_general.ipynb` (added 2 cells at bottom)
- `mjosa_code/notebooks/9_svm_028.ipynb` (added 2 cells at bottom)
- `mjosa_code/notebooks/10_apply_028_model_to_057.ipynb` (added 2 cells at bottom)
- `mjosa_code/notebooks/11_apply_057_model_to_028.ipynb` (added 2 cells at bottom)

### CSV Outputs (created when you run analysis cells):
- `./saved_data/classification_summary_057_on_057.csv`
- `./saved_data/classification_summary_028_crop_1.csv`
- `./saved_data/classification_summary_028_crop_2.csv`
- `./saved_data/classification_summary_028_crop_3.csv`
- `./saved_data/classification_summary_028_crop_4.csv`
- `./saved_data/classification_summary_028_to_057.csv`
- `./saved_data/classification_summary_057_to_028_latest_crop.csv`

## Next Steps

1. **Run all 4 notebooks** from top to bottom
2. **Run the analysis cells** at the bottom of each notebook
3. **Compare the CSV outputs** side-by-side
4. **Use the metrics** to build your narrative:
   - Tables: use CSV data
   - Figures: save the plots (right-click → save image)
   - Text: use the printed statistics and percentages
   - Conclusions: compare baseline vs cross-transfer performance

## Questions?

If you need:
- Different visualizations (e.g., heatmaps, scatter plots)
- Additional metrics (e.g., spatial autocorrelation, texture analysis)
- Combined analysis across all 4 notebooks
- Different CSV export formats

Just ask! The analysis function is modular and easy to extend.

---

**Implementation Date**: November 15, 2025  
**Status**: ✅ Complete and ready to use
