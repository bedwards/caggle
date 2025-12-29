# Caggle - Kaggle Competition Project

**Author:** Brian Edwards (brian.mabry.edwards@gmail.com)
**Location:** Waco, Texas, USA
**Built with:** [Claude Code](https://claude.com/claude-code)

## Security

- **NEVER commit the Kaggle API token** - stored at `~/.kaggle/kaggle.json`
- Do not add credentials, API keys, or secrets to version control
- `data/` and `submissions/` are gitignored

## Machine

- **Mac Studio M1** (likely M1 Max/Ultra)
- **CPU cores**: 23
- **GPU**: MPS (Metal Performance Shaders) - PyTorch only
- Tree-based models (LightGBM, XGBoost, CatBoost) run on CPU only

## Project Structure

```
caggle/
├── data/                    # gitignored - competition datasets
│   └── <competition-name>/
├── submissions/             # gitignored - submission CSVs
│   └── <competition-name>/
├── src/                     # shared utility modules
│   ├── data.py              # data loading utilities
│   └── models.py            # training utilities (CV, etc.)
├── notebooks/               # Jupyter notebooks for exploration
└── <competition-name>.py    # competition-specific scripts
```

## Tech Stack (Updated Dec 2025)

| Package | Version | Purpose |
|---------|---------|---------|
| autogluon | 1.5.0 | AutoML (best-in-class tabular) |
| lightgbm | 4.6.0 | Gradient boosting |
| xgboost | 3.1.2 | Gradient boosting |
| catboost | 1.2.8 | Gradient boosting |
| pytorch-tabular | 1.1.1 | Neural nets for tabular (TabNet, etc.) |
| pytorch-tabnet | 4.1.0 | TabNet |
| optuna | 4.6.0 | Hyperparameter tuning |
| torch | 2.9.1 | Deep learning (MPS acceleration) |
| scikit-learn | 1.7.2 | ML utilities |
| polars | 1.36.1 | Fast dataframes |
| pandas | 2.3.3 | Dataframes |
| numpy | 1.26.4 | Numerical computing |

## Workflow

```bash
# Download competition data
kaggle competitions download -c <competition-name> -p data/<competition-name>/
unzip data/<competition-name>/<competition-name>.zip -d data/<competition-name>/

# Submit predictions
kaggle competitions submit -c <competition-name> \
  -f submissions/<competition-name>/submission.csv \
  -m "description"

# Check leaderboard
kaggle competitions leaderboard -c <competition-name>
```

## Kaggle CLI Limitations (as of 2025)

- **No discussion forum access** via CLI - use Claude Chrome extension
- API supports: competitions, datasets, kernels, models
- API does NOT support: discussions, comments, user profiles

## Code Style

- Use type hints
- Keep notebooks clean - move reusable code to `src/`
- Version control experiments with clear commit messages

---

## Active Competitions

### playground-series-s5e12 (Diabetes Prediction)

| Property | Value |
|----------|-------|
| Task | Binary classification |
| Target | `diagnosed_diabetes` |
| Metric | AUC-ROC |
| Deadline | Dec 31, 2025 |
| Train rows | 700,000 |
| Test rows | 300,000 |

**Features (25 total):**
- Numerical (15): age, alcohol_consumption_per_week, physical_activity_minutes_per_week, diet_score, sleep_hours_per_day, screen_time_hours_per_day, bmi, waist_to_hip_ratio, systolic_bp, diastolic_bp, heart_rate, cholesterol_total, hdl_cholesterol, ldl_cholesterol, triglycerides
- Categorical (6): gender, ethnicity, education_level, income_level, smoking_status, employment_status
- Binary (3): family_history_diabetes, hypertension_history, cardiovascular_history

#### Competition Strategy (from Discussion Forums)

**Critical Insight: Distribution Shift**
- First ~678K training rows have different causal relationships than tail/test
- **Last ~22K samples match test distribution** - use for validation
- Adversarial validation: last 22K vs test has AUC ~0.50 (indistinguishable)

**Validation Strategy:**
```python
# Use last 22K samples as holdout validation
train_main = train.iloc[:-22000]
train_val = train.iloc[-22000:]
```

**Top Approaches (CV ~0.703-0.705):**
1. GBDT ensemble: XGBoost + LightGBM + CatBoost
2. Neural networks: TabM, RealMLP for diversity
3. Weighted refit: Higher sample weights for last 22K during final training
4. Ensembling: Hill climbing or ridge regression for weight optimization

**Feature Engineering (use cautiously):**
- Target encoding on numerical cols with nunique > 2 (not binary)
- Use original dataset as reference for mean/count encoding
- Bigram/interaction features with target encoding
- Heavy FE may improve CV but hurt LB (distribution shift)

**Original Dataset:**
- Source: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- Key predictors removed: HbA1c, glucose_fasting, glucose_postprandial, diabetes_risk_score
- Original achieves AUC ~0.9999; competition data ~0.73
- Best use: Reference for statistical features only (risky to concatenate)

**Target Scores:**
- Top public LB: ~0.708 (1st: 0.70804)
- Competitive: 0.707+ (top 50), 0.706+ (top 200)
- Strong CV on last 22K: ~0.705+

**Warnings:**
- Don't trust full-data CV (wrong distribution)
- CV-LB gap common with aggressive FE
- Shakeup risk: public/private split may differ
- Don't over-optimize to public LB

#### Submissions

| Date | Model | CV AUC | LB AUC | Notes |
|------|-------|--------|--------|-------|
| 2025-12-29 | LightGBM baseline | 0.72688 | TBD | 5-fold CV on all data |

**Resources:**
- [Competition page](https://www.kaggle.com/competitions/playground-series-s5e12)
- [Discussion forums](https://www.kaggle.com/competitions/playground-series-s5e12/discussion)

---

## Reading Kaggle Discussions

Use Claude Chrome extension with this prompt:

```
I'm working on the Kaggle [COMPETITION NAME] competition.

Please read this discussion page and summarize:
1. What are the top approaches people are using?
2. Any useful feature engineering tips?
3. Is there an original dataset this synthetic data is based on?
4. What CV scores are competitive (to aim for)?
5. Any common pitfalls or warnings about leaderboard shakeup?

Format as a concise bulleted summary I can share with my coding assistant.
```

## Kaggle ToS Notes

Per ToS;DR: "Spidering, crawling, or accessing the site through any automated means is not allowed" - but Kaggle provides an official API. The restriction is about unauthorized mass scraping, not authenticated personal use.
