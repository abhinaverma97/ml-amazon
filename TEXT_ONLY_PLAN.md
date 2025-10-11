# Text-only GPU-based Modeling Plan (optimize SMAPE)

Goal: build a text-only model (GPU-accelerated) that minimizes SMAPE on product price prediction using only textual fields from `catalog_content` (item name, bullet points, product description, IPQ). Follow constraints: do not use sentence-transformers, no hyperparameter tuning stage in this plan. Focus is model building & feature engineering for best SMAPE.

Summary contract
- Inputs: training CSV rows with columns: `sample_id`, `catalog_content`, `image_link`, `price` (train); test rows without `price`.
- Outputs: predictions: CSV of `sample_id, price` with positive floats.
- Failure modes: missing parsed fields, malformed text, extreme outliers causing SMAPE blowups. Use sensible fallbacks (defaults, clipping) during feature construction.

High-level approach (why this will lower SMAPE)
- Use strong GPU-accelerated transformer encoder(s) on raw and cleaned text to capture semantics and pricing-related modifiers (brand, pack size, adjectives like "organic", "premium").
- Combine learned transformer embeddings with engineered numeric features (IPQ normalized, item/description lengths, bullet counts, price-related keyword flags).
- Train a robust regression head (two options: 1) a light dense MLP head on transformer embeddings with robust loss; 2) a LightGBM/NGBoost on concatenated embeddings + numeric features — but preferred is a full end-to-end GPU model to exploit GPU (option 1).
- Use log-target training (predict log(price+1)) to stabilize training and reduce relative errors; predict by inverse transform to produce final price.
- Optimize loss approximating SMAPE indirectly: use Huber or log-cosh on log-target; additionally monitor SMAPE and optionally add SMAPE surrogate loss terms if needed.

Feature engineering (text-only)
1. Parse `catalog_content` into structured parts (deterministic, regex based):
   - `item_name` (title)
   - `bullet_points` (concatenate as single field or keep as sequence)
   - `product_description`
   - `ipq_value`, `ipq_unit`
2. Normalize IPQ units and produce numeric `ipq_value_norm` if possible. If ambiguous, keep `ipq_value` as a numeric feature and a unit categorical feature.
3. Derived numeric features:
   - `catalog_char_len`, `item_name_char_len`, `description_char_len`
   - `catalog_word_count`, `item_name_word_count`, `description_word_count`
   - `bullet_count` (number of bullet points)
   - `has_brand_token` flag (detect known brand tokens via frequency or short list built from training data)
   - `premium_kw_flag` (presence of keywords like premium, organic, deluxe, gourmet, handcrafted)
4. Create an input text strategy (several alternatives to feed transformer):
   - Strategy A (single-field concat): [ITEM_NAME] + " [SEP] " + bullet_points + " [SEP] " + description + " [SEP] " + "IPQ: <value> <unit>".
   - Strategy B (dual input): encoder on `item_name` and another on `full_description` then pool and concat embeddings before head.
   - Strategy C (title-first with windowing): put title and first N tokens of description; include other fields as short appended tokens. This helps when text exceeds model input length.

Model choices (GPU, no sentence-transformers)
- Backbone (GPU-accelerated transformer encoder):
  - DistilBERT / BERT / RoBERTa family (Hugging Face): use pretrained base models (cased/uncased depending on normalization) and fine-tune.
  - For efficiency, prefer `roberta-base` or `distilroberta-base` (GPU-friendly). Avoid large models >8B.

- Regression head (two variants):
  1. End-to-end fine-tuned transformer + MLP head
     - Pool encoder outputs with mean/max + CLS combination
     - Concatenate engineered numeric features
     - Small MLP (2 layers, e.g., 512 -> 128) with dropout and final single output neuron predicting log(price+1)
     - Loss: MSE on log-target or Huber on log-target. Monitor SMAPE on validation and add SMAPE-weighted penalty if gap persists.
  2. Two-stage: transformer embeddings (frozen or fine-tuned) → extract embeddings for dataset → train a gradient-boosted tree (LightGBM/HistGradientBoosting) on embeddings+features
     - This is useful if GPU memory or fine-tuning budget is limited. But final recommendation prioritizes option 1 for best performance on GPU.

Loss, target transform, and evaluation
- Predict y' = log(1 + price). Train with loss = MSE(y', y'_pred) or Huber on y'.
- At inference convert: price_pred = exp(y'_pred) - 1; clip at small positive epsilon (e.g., 0.01).
- While training, compute SMAPE on the validation fold (not as optimization target) and use it as the main selection metric.
- Optionally explore using a differentiable surrogate for SMAPE (e.g., smoothed SMAPE) as an auxiliary loss if MSE-log doesn't reduce validation SMAPE sufficiently.

Validation strategy (to robustly estimate SMAPE)
- Use K-Fold CV (K=5) stratified by price bins (e.g., quantile buckets) to ensure both low and high-price examples are in each fold.
- For each fold: train on 4 folds, validate on 1. Save fold predictions for ensemble.
- Report fold-wise SMAPE and aggregate.
- Use early stopping by monitoring validation SMAPE or validation log-loss to prevent overfit.

Baseline experiments (quick wins)
1. Simple baseline: TF-IDF (title + description) -> LightGBM (train on log(price+1)) → measure SMAPE. This provides a strong classical baseline and is cheap.
2. Transformer baseline (single field concat) fine-tuned end-to-end with MSE-log loss.
3. Transformer + numeric features (concatenate) — expected improvement.
4. Ensembling: average predictions from different backbones (roberta-base, distilroberta) and different input strategies.

Training recipes and practical tips (GPU-focused)
- Mixed precision (FP16) training to increase throughput (use PyTorch AMP or Trainer fp16 flag).
- Use gradient accumulation if batch size is constrained by GPU RAM.
- Use AdamW optimizer with a small learning rate for fine-tuning (e.g., 2e-5) but note: no HPO now — use reasonable defaults.
- Use batch size as large as memory allows; larger batch sizes stabilize embeddings for regression.
- Periodically compute SMAPE on validation fold (every epoch or every N steps) for early stopping.

Postprocessing
- Convert predictions back from log space and clip to a minimum positive value (0.01).
- Consider light ensemble (fold-average) to reduce variance.
- Optionally apply small monotonic isotonic calibration or quantile correction if distributional shift observed vs validation.

Error analysis and focused improvements
- Slice evaluation: compute SMAPE per category, per IPQ range, and per price-decile. Identify segments with high SMAPE and engineer features for them.
- Use SHAP/feature attribution on final model input (via integrated gradients or attention attribution) to identify which tokens/features drive predictions.
- Add price-range specialized heads: e.g., train separate small heads for low/medium/high price buckets.

Deliverables (what to produce in code next)
- Data parsing and feature extraction module for text fields.
- TF-IDF+LightGBM baseline notebook and results (quick SMAPE number).
- Full transformer training script (PyTorch Lightning or HF Trainer) that:
  - tokenizes according to chosen input strategy
  - supports mixed precision, gradient accumulation
  - concatenates numeric features to the head
  - saves fold predictions and model checkpoints
- Evaluation notebook computing SMAPE per slice and summary.

Risks and mitigations
- Extreme outliers (very large prices) dominate SMAPE: mitigate with log-target training and stratified folds.
- IPQ parsing errors: create robust fallback (e.g., NaN -> 1.0) and a unit "unknown" category.
- Very long `catalog_content` texts: use windowing and prioritize item title + leading description; experiment with dual-encoder splits.

Next steps (implementation order, no HPO yet)
1. Implement deterministic parser for `catalog_content` and create cleaned training and test CSVs with extracted fields.
2. Implement TF-IDF baseline with LightGBM on log-target and compute validation SMAPE.
3. Implement transformer end-to-end training script (roberta-base or distilroberta-base) following Strategy A; train with K=3 folds first for speed.
4. Add numeric features to transformer head and retrain.
5. Ensemble folds and compute final CV SMAPE.

If you want, I can now create the `TEXT_ONLY_PLAN.md` file in the repo (this is the plan) and also scaffold the first parser + TF-IDF baseline script next.