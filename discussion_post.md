# Building This Solution with Claude Code: A Transparent Walkthrough

I built this entire solution using [Claude Code](https://claude.com/claude-code), Anthropic's command-line interface (CLI) for their AI assistant Claude. This post describes exactly what that process looked like, including what worked and what didn't.

## What is Claude Code?

Claude Code is a terminal-based AI assistant that can read files, run commands, write code, and search the web. You describe what you want, and it does the work while you watch (or go get coffee).

## The Session

Here's what happened in our conversation:

### 1. Finding a Competition

I asked Claude to find a good active Kaggle competition. It ran `kaggle competitions list`, searched the web for details, and recommended Playground Series S5E12 (Diabetes Prediction) because:
- Tabular data (good for gradient boosting)
- Active community (3,700+ teams)
- Deadline soon (forces fast iteration)

### 2. Reading Discussions

The Kaggle API doesn't support reading discussion forums. Claude tried several approaches:
- Web scraping (blocked by CAPTCHA)
- Unofficial APIs (none exist)

We ended up using the Claude Chrome extension. I opened the discussion page in my browser, pasted a prompt Claude wrote, and got a summary of community insights.

**Key insight discovered:** The first ~678,000 training samples have a different distribution than the test set. Only the last ~22,000 samples match. Using all data for cross-validation gives misleading scores.

### 3. First Baseline

Claude wrote a LightGBM script with 5-fold cross-validation. Results:
- CV AUC: 0.727 (looked great!)
- LB AUC: 0.696 (reality check)

The gap was exactly what the discussions predicted.

### 4. Fixing Validation

Claude rewrote the code to use the last 22,000 samples as a holdout validation set. New results:
- Val AUC: 0.696 (matches LB)
- Now we have a reliable signal

### 5. Ensemble

Claude built an ensemble of three gradient boosting models:
- LightGBM
- XGBoost
- CatBoost

Used Ridge regression to find optimal weights. Final validation AUC: 0.696.

## What Worked Well

1. **Fast iteration.** Claude wrote and ran code faster than I could type.
2. **No context switching.** Everything happened in one terminal session.
3. **Good research skills.** It found relevant discussions and extracted key insights.

## What Didn't Work

1. **Can't access Kaggle discussions via API.** Required manual browser workaround.
2. **Package conflicts.** Some libraries had version incompatibilities that needed fixing.
3. **No GPU access.** Running locally on CPU meant slower training.

## The Actual Code

The full project is public: https://github.com/bedwards/caggle

It includes:
- `s5e12_ensemble.py` - Main ensemble script
- `playground-series-s5e12.py` - Original baseline
- `src/` - Shared utility functions
- `notebooks/` - Educational notebook
- `CLAUDE.md` - Project documentation (also written by Claude)

## Honest Assessment

Using Claude Code saved me several hours of boilerplate. I didn't have to:
- Look up API documentation
- Debug basic syntax errors
- Write data loading code
- Format submission files

I still had to:
- Provide the competition insight (via discussion summary)
- Make strategic decisions about what to try
- Verify the code actually worked

The distribution shift insight came from human competitors sharing their analysis. Claude found and applied it, but didn't discover it independently.

## Numbers

| Model | Val AUC (last 22K) |
|-------|-------------------|
| LightGBM | 0.696 |
| XGBoost | 0.693 |
| CatBoost | 0.696 |
| Weighted Ensemble | 0.696 |

Not world-beating, but honest. The ceiling for this dataset appears to be around 0.708 (current top of leaderboard).

---

Questions welcome. I'll try to answer based on what I actually observed in the session.
