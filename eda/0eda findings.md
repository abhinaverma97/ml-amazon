# Comprehensive Dataset Understanding

## Dataset Overview

**Challenge**: Smart Product Pricing - Predict product prices using multimodal data (text + images)

**Data Size**: 
- Training: 75,000 samples
- Test: 75,000 samples
- Sample test: 100 samples

**Data Quality**: Excellent - No missing values in any field

## Data Structure

### Features
1. **sample_id** (int64): Unique identifier (0-299,439 range)
2. **catalog_content** (object): Rich text containing product information
3. **image_link** (object): Amazon product image URLs
4. **price** (float64): Target variable (training only)

### Target Variable Analysis
- **Range**: $0.13 - $2,796.00
- **Mean**: $23.65
- **Median**: $14.00  
- **Distribution**: Right-skewed with outliers (7.37% outliers)
- **Key Percentiles**:
  - 25th: $6.79
  - 75th: $28.62
  - 95th: $75.71
  - 99th: $145.25

## Catalog Content Structure

### Content Components (Structured Format)
1. **Item Name** (100% coverage): Product title
2. **Bullet Points** (81% coverage): 0-10 bullet points (mostly 5)
3. **Product Description** (43.4% coverage): Detailed description
4. **Value** (100% coverage): Item Pack Quantity (IPQ)
5. **Unit** (100% coverage): Measurement unit

### Content Statistics
- **Length**: 32-7,894 characters (mean: 909)
- **Word Count**: 7-1,333 words (mean: 148)
- **Correlation with Price**: Weak positive (r=0.14-0.15)

### Value/Unit Analysis (IPQ - Item Pack Quantity)
**Most Common Values**:
- 1.0 (8.8%), 16.0 (5.8%), 12.0 (4.2%), 24.0 (3.1%), 8.0 (2.9%)
- Range: 0-10,752 (mean: 51.5)

**Most Common Units**:
- Ounce (54.6%), Count (23.2%), Fl Oz (14.7%), ounce (2.7%)

**Price Correlation**: Weak positive (r=0.12)

## Product Categories (Estimated)

Based on item names, products fall into:
1. **Food & Beverages** (52.4%) - Mean price: $26.34
2. **Other** (38.3%) - Mean price: $21.82
3. **Clothing & Accessories** (3.3%) - Mean price: $19.30
4. **Health & Beauty** (2.7%) - Mean price: $28.77
5. **Home & Garden** (1.5%) - Mean price: $20.90
6. **Pet Supplies** (1.0%) - Mean price: $38.90
7. **Baby & Kids** (0.5%) - Mean price: $22.44

## Image Data

- **Coverage**: 100% of records have image links
- **Source**: All from `m.media-amazon.com` (Amazon product images)
- **URL Length**: Consistent 51 characters
- **Format**: Standard Amazon image URLs

## Sample Records Examples

### Record 1: Food Product
- **Price**: $12.20
- **Product**: Log Cabin Sugar Free Syrup, 24 FL OZ (Pack of 12)
- **Value**: 12.0, Unit: Count
- **Features**: 5 bullet points, detailed nutritional info

### Record 2: Health Product  
- **Price**: $38.54
- **Product**: Raspberry Ginseng Oolong Tea (50 tea bags, ZIN: 543034) - 2 Pack
- **Value**: 2.0, Unit: Count
- **Features**: Long detailed description, ingredient lists

### Record 3: Food Condiment
- **Price**: $17.86  
- **Product**: Walden Farms Honey Dijon Dressing (12 oz Bottle x 2)
- **Value**: 2.0, Unit: Count
- **Features**: Health-focused (calorie-free, keto-friendly)

## Key Insights for Modeling

### Text Features to Extract
1. **Product Name/Title** - Primary identifier
2. **Brand Information** - Often in item name
3. **Product Category** - Food, health, etc.
4. **Quantity/Size** - From value/unit fields
5. **Key Attributes** - From bullet points (organic, sugar-free, etc.)
6. **Descriptive Features** - Length, word count, sentiment

### Numeric Features
1. **IPQ Value** - Item pack quantity
2. **Content Length** - Character/word count
3. **Bullet Point Count** - Number of features listed
4. **Price-relevant Keywords** - Premium, organic, etc.

### Image Features (To Extract)
1. **Visual Product Category** - Food, electronics, etc.
2. **Package Type** - Bottle, box, bag
3. **Color Scheme** - Brand colors
4. **Text in Images** - Brand names, labels
5. **Product Size Indicators** - Visual cues

### Modeling Considerations

1. **Target Transformation**: Consider log transformation due to right skew
2. **Outlier Handling**: 7.37% outliers need careful treatment for SMAPE
3. **Feature Engineering**: Combine text and image features effectively
4. **Multimodal Approach**: Text (NLP) + Image (CV) fusion
5. **Evaluation Metric**: Optimize for SMAPE (symmetric mean absolute percentage error)

### Data Quality Notes
- **Excellent Quality**: No missing values
- **Consistent Format**: Structured text format
- **Reliable Images**: All from Amazon, accessible
- **No Data Leakage**: No overlap between train/test sample IDs
- **Balanced Distribution**: Good variety of products and price ranges

## Recommended Modeling Pipeline

1. **Text Processing**: Extract structured components, NLP embeddings
2. **Image Processing**: Download images, extract visual features
3. **Feature Engineering**: Combine multimodal features
4. **Model Selection**: Regression models optimized for SMAPE
5. **Ensemble Methods**: Combine text and image model predictions
6. **Validation Strategy**: Cross-validation with SMAPE metric

This dataset provides rich multimodal information suitable for building a comprehensive product pricing model that leverages both textual product descriptions and visual product images.

## EDA Visualizations and Key Findings

The repository includes an EDA script (`comprehensive_eda.py`) which produced several plots saved under the `eda/` folder. I ran the script against `dataset/train.csv` and `dataset/test.csv` and the key findings (with filenames) are summarized below.

1. Price distribution and outliers (file: `eda/price_distribution_analysis.png`)
  - Price statistics (train): count=75,000, mean=$23.65, std=$33.38, min=$0.13, median=$14.00, max=$2796.00
  - Distribution is strongly right-skewed; log-transform is recommended for modelling.
  - Outlier analysis (IQR rule): lower bound ≈ -$25.95, upper bound ≈ $61.37 → 5,524 outliers (≈7.37% of train).

2. Text feature summaries and visuals (file: `eda/text_features_analysis.png`, `eda/item_names_wordcloud.png`)
  - `catalog_content` length: mean ≈ 909 characters (range ≈ 32–7,894).
  - `item_name` length: mean ≈ 127 characters.
  - `product_description` length: mean ≈ 280 characters (many records have short or missing descriptions).
  - Bullet points: mean ≈ 3.49 bullets per product (most products list 0–5 bullets).
  - Word cloud (`eda/item_names_wordcloud.png`) highlights common product words useful for category/brand extraction.

3. IPQ (Item Pack Quantity) analysis (part of text features)
  - IPQ coverage: ~74,060 values parsed in train (some parsing variations exist due to inconsistent units formatting).
  - Top IPQ values: 1.0 (6,756), 16.0 (4,016), 12.0 (3,605), 8.0 (2,172), 24.0 (2,076).
  - IPQ units show inconsistency (examples: `Ounce`, `ounce`, `oz`, `Fl Oz`, `fl oz`, `Count`, `count`). Normalization required.

4. Feature correlations and scatter plots (files: `eda/correlation_matrix.png`, `eda/scatter_plots_correlations.png`)
  - Weak positive correlations observed between price and textual size/quantity features (correlation coefficients ~0.1–0.15).
  - No single text-length or IPQ feature explains price variance — a multimodal approach is justified.

5. Simple keyword-based category analysis (files: `eda/category_distribution.png`, `eda/price_by_category.png`)
  - Categories (basic heuristics) and price summary (means):
    - Beverages: count=19,541, mean ≈ $28.57
    - Snacks & Sweets: count=14,174, mean ≈ $22.83
    - Condiments & Spices: count=6,373, mean ≈ $18.81
    - Personal Care: count=1,357, mean ≈ $21.65
    - Clothing: count=552, mean ≈ $17.80
    - Other: count=33,003, mean ≈ $22.20
  - Category-level boxplots show different price spreads; consider category-specific preprocessing or models.

6. Train vs Test distribution checks (file: `eda/train_test_comparison.png`)
  - Train vs test distributions for text-length features are very similar (means differ by <4 characters on average).
  - `ipq_value` mean: train ≈ 54.31, test ≈ 58.31 — similar but test has larger std (some extreme IPQ values present).
  - Overall, no major dataset shift detected in basic features; still validate downstream when training models.

7. Image URL analysis
  - All image links are from `m.media-amazon.com` and have identical length (51 characters). No missing image links.
  - Filenames are consistent and appear to be valid Amazon product image URLs — proceed to download and extract CV features.

Next steps (recommended)
- Normalize IPQ units (map `Ounce`, `ounce`, `Oz`, `oz` → `oz`, and harmonize `Fl Oz` variants) and convert to a single numeric representation per product (e.g., normalized package volume/weight when applicable).
- Parse and standardize brand and pack-size tokens from `item_name` (use regex + gazetteers), and build a clean `category` column using a trained classifier instead of simple heuristics.
- Create TF-IDF and embedding features from `item_name`, `bullet points`, and `product_description`.
- Download images using the provided `download_images` utility and extract visual embeddings (EfficientNet/ResNet or a smaller CLIP-style model within the parameter limit).
- Train baseline models (log-target regression + tree-based models e.g., LightGBM) and an image+text fusion model. Optimize directly for SMAPE and validate using cross-validation.
- Carefully handle price outliers (consider clipping, robust losses, or specialized validation folds) because SMAPE penalizes relative errors.

If you want, I can now:
- (A) Update `DATASET_UNDERSTANDING.md` further with inline links to the actual PNG files stored under `eda/`.
- (B) Add a `requirements.txt` and small helper notebook that downloads images and demonstrates extracting 1–2 sample image embeddings.
