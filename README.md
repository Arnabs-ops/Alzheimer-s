# Alzheimer's Disease Detection & Analysis

## Project Overview

This project focuses on building machine learning models to support early detection, progression forecasting, and interpretability of Alzheimer's Disease risk using real genomic variant data from Alzheimer's Disease Variant Portal (ADVP).

## Goals

- **Early Detection**: Develop models to identify Alzheimer's Disease in its early stages
- **Progression Forecasting**: Predict disease progression patterns
- **Interpretability**: Understand what factors contribute to disease risk

## Project Structure

```
Alzhemiers/
├── data/                    # Dataset storage
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned and preprocessed data
│   └── external/           # External reference data
├── notebooks/              # Jupyter notebooks
│   └── alzheimer_analysis.ipynb
├── src/                    # Source code modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── visualization.py
├── results/                # Model outputs and visualizations
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup Instructions

1. **Clone/Download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download datasets** to the `data/raw/` directory
4. **Run the notebook**:
   ```bash
   jupyter notebook notebooks/alzheimer_analysis.ipynb
   ```

## Dataset Information

- **Source**: Alzheimer's Disease Variant Portal (ADVP) genomic data
- **Files**: 
  - `advp.hg38.tsv`: Genomic variants with positions and associations
  - `advp.hg38.bed`: Genomic regions in BED format
- **Content**: 6,347+ genomic variants associated with Alzheimer's Disease
- **Features**: P-values, gene associations, study types, phenotypes, genomic consequences
- **Privacy**: All data is de-identified and compliant with usage guidelines

## Analysis Features

- **Genomic Analysis**: Manhattan plots, chromosome distributions, gene associations
- **Machine Learning**: Multiple algorithms for variant significance prediction
- **Interpretability**: SHAP analysis, feature importance, phenotype associations
- **Visualization**: Interactive plots, statistical summaries, clinical insights

## Contributing

This is a hackathon project for educational purposes. Please ensure all data usage complies with the provided guidelines and maintains patient privacy.

## License

Educational use only - please respect data usage agreements.
