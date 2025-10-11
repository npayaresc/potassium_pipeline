# Data Analysis Summary - Potassium Pipeline

**Analysis Date:** 2025-10-08
**Reports Location:** `reports/`

---

## 📊 Executive Summary

### Key Findings

✅ **NEW data has 1,062 spectral files** (894 from newdata + 168 from OneDrive)
✅ **NEW reference file has 307 labeled samples** with potassium values
❌ **CRITICAL ISSUE: Zero matches between reference IDs and raw data filenames**
⚠️ **NEW data has significantly higher potassium concentrations** (mean: 6.86% vs 4.79%)

---

## 1. Reference Data Analysis

### OLD Reference File: `Final_Lab_Data_Nico_New.xlsx`
- **Total samples:** 3,169
- **Columns:** 16 (Sample ID + multiple elements)
- **Potassium column:** "Potassium dm"
- **All samples have K values:** 3,169/3,169 (100%)

**Potassium Distribution (OLD):**
```
Range:        0.009 - 19.000%
Mean ± Std:   4.788 ± 3.028%
Median:       4.280%
Q1 / Q3:      2.520% / 6.450%
```

**Concentration Ranges (OLD):**
- Zero values: 0
- Very low (0-0.1%): 1 sample
- Low (0.1-0.5%): 52 samples
- Medium (0.5-2.0%): 633 samples
- **High (2.0%+): 2,483 samples** ← Majority of samples

### NEW Reference File: `Lab_data_updated_potassium2.xlsx`
- **Total samples:** 503
- **Columns:** 19 (includes S, Mo, Zn, P, Fe, B, Mn, Mg, Ca, Cu, Na, K, NO3, N, C)
- **Potassium column:** "K 766.490 (wt%)" ← Uses K I spectral line wavelength!
- **Samples with K values:** 307/503 (61%)
- **Missing K values:** 196 samples

**Potassium Distribution (NEW):**
```
Range:        0.251 - 18.916%
Mean ± Std:   6.859 ± 2.930%
Median:       7.349%
Q1 / Q3:      6.038% / 8.615%
```

**Concentration Ranges (NEW):**
- Zero values: 0
- Very low (0-0.1%): 0 samples
- Low (0.1-0.5%): **35 samples** ← Important for low-concentration modeling
- Medium (0.5-2.0%): 2 samples
- **High (2.0%+): 270 samples** ← 88% of labeled samples!

### NEW Reference File v1: `Lab_data_updated_potassium.xlsx`
- **Total samples:** 453
- **Columns:** 2 (Sample No., Sample ID only - NO POTASSIUM VALUES)
- ⚠️ This file appears to be incomplete - no concentration data

---

## 2. Raw Spectral Data Analysis

### OLD Data: `data/raw/data_5278_Phase3/`
```
Total files:        4,642
Unique samples:     2,112
Shots per sample:   ~2.2 (average)
Wavelength points:  ~2,514 (estimated from NEW data)
```

**Sample ID Format:**
```
1053789KENP1S001_G_2025_01_04_1.csv.txt
└─────┬────────┘
   Sample ID
```

### NEW Data: `data/raw/newdata/`
**Subdirectories by date:**
```
23-09-2025:  112 files
24-09-2025:  304 files  ← Largest batch
25-09-2025:   72 files
29-09-2025:  174 files
30-09-2025:  232 files
─────────────────────
Total:       894 files
```

**Sample ID Format (NEW):**
```
MPN_0000_002025_CNGS7330000001_POT_0000_S00531_P_Y_27062025_1000_0_01.csv.txt
└─────────────────────────────────────────────────────────┬──────────────────┘
                    Complex identifier including sample number (S00531)
```

### NEW Data: `data/raw/OneDrive_1_9-22-2025/`
**Subdirectories by date:**
```
16-09-2025:  48 files
18-09-2025:  84 files
19-09-2025:  36 files
────────────────────
Total:      168 files
```

**Total NEW spectral files: 1,062** (894 + 168)

---

## 3. ⚠️ CRITICAL ISSUE: Sample ID Mismatch

### The Problem
**Zero matches between reference sample IDs and raw data filenames!**

**Reference IDs look like:**
```
BTH/25/09/08/001
BTH/25/09/08/002
...
```

**Raw data IDs look like:**
```
MPN_0000_002025_CNGS7330000001_POT_0000_S00531_P_Y_27062025_1000_0_01.csv
MPN_0000_002025_CNGS7330000001_POT_0000_S02068_P_Y_03072025_1000_01.csv
...
```

### What This Means
1. **Cannot directly link reference concentrations to spectral files**
2. **Need a mapping file or key to match samples**
3. **Possible that:**
   - Sample numbers are encoded differently (e.g., "S00531" ↔ "BTH/25/09/08/001")
   - Reference file uses different naming convention
   - Files are from different batches/experiments

### Action Required
📋 **You need to provide:**
- A mapping file that links reference IDs to raw data filenames
- OR clarification on how to extract matching IDs from the filenames
- OR confirm that "Sample ID" in reference doesn't match "Raw files_Sample ID" column

---

## 4. Data Quality Comparison: OLD vs NEW

### Sample Size
```
                    OLD         NEW      Difference
Labeled samples:    3,169       307      -2,862 (-90%)
Spectral files:     4,642     1,062      -3,580 (-77%)
```

### Potassium Concentration Distribution
```
Metric          OLD         NEW      Difference
─────────────────────────────────────────────────
Mean            4.788%      6.859%   +2.071% (+43%)
Median          4.280%      7.349%   +3.069% (+72%)
Std Dev         3.028%      2.930%   -0.098% (-3%)
Min             0.009%      0.251%   +0.242%
Max            19.000%     18.916%   -0.084%
```

### Key Observations

✅ **NEW data advantages:**
- Higher average concentrations (6.86% vs 4.79%)
- More consistent (lower std dev)
- Better coverage of medium-high range
- **Has 35 samples in critical 0.1-0.5% range**

⚠️ **NEW data concerns:**
- **Much smaller dataset** (307 vs 3,169 samples)
- **Fewer low-concentration samples** (1 vs 53 below 0.5%)
- **Less coverage of concentration extremes** (min: 0.251% vs 0.009%)
- **39% of samples missing K values** (196/503)

⚠️ **Distribution shift:**
- OLD: Broad distribution (0.009-19.0%), mean 4.79%
- NEW: **Narrower, shifted higher** (0.251-18.92%), mean 6.86%
- NEW has **88% of samples above 2.0%** vs 78% in OLD
- **Model trained on NEW may not generalize to low concentrations!**

---

## 5. Spectral File Format

### Successfully Read
- **Format:** Tab-separated values (TSV)
- **Columns:** 2 (wavelength, intensity)
- **Wavelength points:** ~2,514
- **No header row**

### Sample Structure
```
200.00    12.45
200.05    13.21
200.10    14.67
...
```

⚠️ **Minor Issue:** String formatting error when displaying wavelength ranges (non-critical)

---

## 6. Recommendations

### Immediate Actions Required

1. **🔴 PRIORITY: Resolve Sample ID Mismatch**
   - Provide mapping between reference IDs and raw data filenames
   - Check if "Raw files_Sample ID" column in reference matches raw filenames
   - Consider creating a mapping script if pattern is identifiable

2. **📊 Data Quality Assessment**
   - Verify the 35 low-concentration samples (0.1-0.5%) are high quality
   - Check why 196/503 samples are missing K values in NEW reference
   - Consider whether to combine OLD + NEW data for training

3. **⚖️ Model Training Strategy**
   - **Option A:** Use only NEW data (smaller but higher quality, better element coverage)
   - **Option B:** Use OLD data (larger, better low-concentration coverage)
   - **Option C:** Combine both datasets (requires careful validation)
   - **Option D:** Train separate models for different concentration ranges

### Data Combination Strategy (if choosing Option C)

**Advantages:**
- Larger training set (3,169 + 307 = 3,476 samples)
- Better coverage of full concentration range
- More robust for extreme values

**Concerns:**
- Different measurement methods/equipment?
- Reference file formats differ (NEW has multi-element data)
- Distribution shift may cause issues
- Need to verify data compatibility

### Before Training

✅ **Resolve sample ID matching issue**
✅ **Decide on training data source** (OLD, NEW, or combined)
✅ **Validate low-concentration samples** (critical for model performance)
✅ **Check for outliers** in NEW data
✅ **Update config file** to point to correct directories

---

## 7. Visualizations Generated

All plots saved to `reports/`:

1. **`reference_analysis_OLD_Potassium_dm.png`**
   - Distribution, box plot, violin plot
   - CDF, Q-Q plot, concentration ranges

2. **`reference_analysis_NEW_K_766.490_(wtpct).png`**
   - Same analysis for NEW data

3. **`reference_comparison_OLD_vs_NEW.png`**
   - Side-by-side comparison
   - Overlaid histograms and CDFs
   - Statistical summary table

---

## 8. Next Steps

### To Continue with NEW Data:
```bash
# 1. Resolve sample ID mapping first!
# 2. Update config to use NEW data:
# Edit src/config/pipeline_config.py:
#   Line 896: raw_data_dir = "data/raw/newdata"
#   Line 887: reference_data_path = "data/reference_data/Lab_data_updated_potassium2.xlsx"

# 3. Run training after mapping is resolved:
uv run python main.py train --data-parallel --feature-parallel
```

### To Continue with OLD Data (safer option):
```bash
# Already configured, just run:
uv run python main.py train --data-parallel --feature-parallel
```

---

## 📧 Questions for Data Provider

1. **How do we map reference IDs to raw data filenames?**
   - Is there a mapping file?
   - What does "Raw files_Sample ID" column represent?
   - How to extract sample ID from long filename format?

2. **Why are 196/503 samples missing K values in NEW reference?**
   - Are measurements pending?
   - Should these be excluded?

3. **Can OLD and NEW data be combined?**
   - Same equipment/methodology?
   - Same sample preparation?
   - Compatible for joint training?

4. **Are the 35 low-concentration samples in NEW data reliable?**
   - Need to prioritize these for low-range predictions

---

**End of Analysis Report**
