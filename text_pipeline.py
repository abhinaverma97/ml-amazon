import re
import os
import json
import math
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

RANDOM_STATE = 42
N_FOLDS = 5


def read_data(base_path='dataset'):
    train = pd.read_csv(os.path.join(base_path, 'train.csv'))
    test = pd.read_csv(os.path.join(base_path, 'test.csv'))
    return train, test


def parse_catalog_content(df):
    """Extract structured pieces from catalog_content"""
    def extract_ipq(text):
        if pd.isna(text):
            return (np.nan, None)
        v = re.search(r'Value:\s*([0-9.]+)', text)
        u = re.search(r'Unit:\s*([^\n]+)', text)
        val = float(v.group(1)) if v else np.nan
        unit = u.group(1).strip() if u else None
        return val, unit

    def extract_item_name(text):
        if pd.isna(text):
            return ""
        m = re.search(r'Item Name:\s*(.*?)(?=Bullet Point 1:|Product Description:|Value:|$)', text, re.DOTALL)
        return m.group(1).strip() if m else ''

    def extract_description(text):
        if pd.isna(text):
            return ""
        m = re.search(r'Product Description:\s*(.*?)(?=Value:|$)', text, re.DOTALL)
        return m.group(1).strip() if m else ''

    def count_bullets(text):
        if pd.isna(text):
            return 0
        return len(re.findall(r'Bullet Point \d+:', text))

    df = df.copy()
    df['ipq_value'], df['ipq_unit'] = zip(*df['catalog_content'].map(extract_ipq))
    df['item_name'] = df['catalog_content'].map(extract_item_name)
    df['product_description'] = df['catalog_content'].map(extract_description)
    df['bullet_count'] = df['catalog_content'].map(count_bullets)

    # derived lengths
    df['catalog_char_len'] = df['catalog_content'].str.len().fillna(0).astype(int)
    df['item_name_char_len'] = df['item_name'].str.len().fillna(0).astype(int)
    df['desc_char_len'] = df['product_description'].str.len().fillna(0).astype(int)
    df['item_name_wc'] = df['item_name'].str.split().map(lambda x: len(x) if isinstance(x, list) else 0)
    df['desc_wc'] = df['product_description'].str.split().map(lambda x: len(x) if isinstance(x, list) else 0)

    return df


def normalize_ipq_unit(unit):
    if pd.isna(unit):
        return 'unknown'
    u = unit.strip().lower()
    if u in ['ounce', 'oz', 'oz.']:
        return 'oz'
    if 'fl' in u and 'oz' in u:
        return 'fl_oz'
    if u in ['count', 'ct', 'count.']:
        return 'count'
    return u


def build_features(train, test, n_tfidf=50000, n_svd=128):
    # Parse
    train = parse_catalog_content(train)
    test = parse_catalog_content(test)

    # Normalize ipq units
    for df in [train, test]:
        df['ipq_unit_norm'] = df['ipq_unit'].map(normalize_ipq_unit).fillna('unknown')

    # numeric features
    num_features = ['ipq_value', 'catalog_char_len', 'item_name_char_len', 'desc_char_len',
                    'item_name_wc', 'desc_wc', 'bullet_count']

    # fillna for numeric
    for f in num_features:
        train[f] = train[f].fillna(-1)
        test[f] = test[f].fillna(-1)

    # text assembly
    def make_text(row):
        parts = [str(row['item_name']), '\n']
        parts.append(str(row['bullet_count']))
        parts.append('\n')
        parts.append(str(row['product_description']))
        parts.append('\nIPQ: ' + str(row.get('ipq_value', ''))) 
        parts.append(' ' + str(row.get('ipq_unit_norm', '')))
        return ' '.join([p for p in parts if p is not None])

    train['full_text'] = train.apply(make_text, axis=1)
    test['full_text'] = test.apply(make_text, axis=1)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=n_tfidf, ngram_range=(1,2), stop_words='english')
    X_text = tfidf.fit_transform(train['full_text'].fillna(''))
    X_test_text = tfidf.transform(test['full_text'].fillna(''))

    # SVD
    svd = TruncatedSVD(n_components=n_svd, random_state=RANDOM_STATE)
    X_text_svd = svd.fit_transform(X_text)
    X_test_svd = svd.transform(X_test_text)

    # numeric scaler
    scaler = StandardScaler()
    X_num = scaler.fit_transform(train[num_features])
    X_test_num = scaler.transform(test[num_features])

    # concat
    X_train = np.hstack([X_text_svd, X_num])
    X_test = np.hstack([X_test_svd, X_test_num])

    feature_names = [f'tfidf_svd_{i}' for i in range(X_text_svd.shape[1])] + num_features

    # free memory
    del X_text, X_test_text, X_text_svd, X_test_svd
    gc.collect()

    return X_train, X_test, feature_names, tfidf, svd, scaler


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1e-6
    return np.mean(np.abs(y_true - y_pred) / denom)


def train_lgbm(X, y, X_test, ids_test, feature_names, n_folds=N_FOLDS):
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}/{n_folds}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'seed': RANDOM_STATE,
            'learning_rate': 0.05,
            'num_leaves': 127,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
        }

        # use callbacks for early stopping and logging (compatible with different LightGBM versions)
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        test_fold_pred = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds += test_fold_pred / n_folds
        models.append(model)

        fold_smape = smape(np.expm1(y_val), np.expm1(val_pred))
        print(f"Fold {fold+1} SMAPE: {fold_smape:.6f}")

    full_smape = smape(np.expm1(y), np.expm1(oof_preds))
    print(f"Full OOF SMAPE: {full_smape:.6f}")

    # save oof and test preds
    return models, oof_preds, test_preds, full_smape


def run_pipeline():
    Path('artifacts').mkdir(exist_ok=True)

    train, test = read_data()
    X_train, X_test, feature_names, tfidf, svd, scaler = build_features(train, test,
                                                                       n_tfidf=50000, n_svd=128)

    # target transform
    y = np.log1p(train['price'].values)

    models, oof_preds, test_preds, full_smape = train_lgbm(X_train, y, X_test, test['sample_id'].values,
                                                          feature_names)

    # persist artifacts
    pd.DataFrame({'sample_id': train['sample_id'], 'oof_pred_log': oof_preds}).to_csv('artifacts/oof_preds.csv', index=False)
    submission = pd.DataFrame({'sample_id': test['sample_id'], 'price': np.expm1(test_preds)})
    submission['price'] = submission['price'].clip(lower=0.01)
    submission.to_csv('artifacts/test_submission.csv', index=False)

    # save models and transformers
    with open('artifacts/tfidf.vocab.json', 'w', encoding='utf8') as f:
        json.dump(tfidf.vocabulary_, f)

    print('\nPipeline finished. Artifacts saved to artifacts/.')


if __name__ == '__main__':
    run_pipeline()
