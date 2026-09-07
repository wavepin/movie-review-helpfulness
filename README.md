# Predicting Movie Review Helpfulness

Can review text and simple metadata predict which movie reviews receive helpful votes? This NLP project compares seven classifiers on **50,000 Amazon Movies & TV reviews**, explores the relationship between review detail and helpfulness, and tests how results change when accounting for review age and existing votes.

The strongest validation model combines **TF-IDF, engineered features, and class-weighted logistic regression**. With a decision threshold tuned on validation data, it achieves **0.617 average precision and 0.580 F1** on the test set.

> The target is **observed helpfulness**: whether a review received at least one helpful vote. Zero votes can reflect limited exposure, so this is not a direct measurement of a review's intrinsic quality.

**[Explore the complete notebook](output/jupyter-notebook/helpfulness-experiments-complete.ipynb)** — saved outputs include the analysis, model comparisons, plots, precision–recall curves, confusion matrix, and coefficient interpretation.

## Contents

- [Research questions](#research-questions)
- [Data](#data)
- [Reproduce the experiments](#reproduce-the-experiments)
- [Method](#method)
- [Results](#results)
- [Limitations and next steps](#limitations-and-next-steps)
- [Credits and references](#credits-and-references)

## Research questions

1. How well can standard NLP classifiers predict whether a review receives helpful votes?
2. How do review length, language, rating, and metadata relate to observed helpfulness?
3. Do reviews discussing plot, expressing emotion, or combining both show different helpful-vote rates?

The project developed from an initial experiment using only reviews with existing helpful votes into a main experiment retaining the entire 50K sample. The original exposed-only task remains in the notebook as a secondary comparison, alongside an analysis restricted to older reviews.

## Data

The included JSONL file is a 50,000-review subset of the **Amazon Reviews 2023 Movies_and_TV** corpus from [McAuley Lab](https://amazon-reviews-2023.github.io/). Each line is one review. The subset was prepared using reservoir sampling with a fixed seed before vote-based filtering; the final notebook starts from this saved sample. It does not recreate sampling from the full corpus, and the original sampling implementation is not included.

| Property | Included subset |
| --- | ---: |
| Reviews | 50,000 |
| Zero helpful votes | 37,346 (74.69%) |
| At least one helpful vote | 12,654 (25.31%) |
| Mean title + body length | 46.52 words |
| Verified purchases | 79.39% |

The notebook compares the sample with stored full-corpus reference statistics from earlier project work. Those reference values are constants, not recalculated from a full-corpus download during execution.

| Input field | Use |
| --- | --- |
| `title`, `text` | Combined as the review's text input; missing values become empty strings |
| `helpful_vote` | Creates the target; never used as an input feature |
| `rating` | Numeric star-rating feature |
| `verified_purchase` | Binary metadata feature |
| `timestamp` | Unix time in milliseconds, converted to review year |
| `asin`, `parent_asin`, `user_id`, `images` | Preserved in the source sample; not used as model features |

All 50,000 rows have usable combined text. The dataset is included unchanged, so no download, API key, or resampling is required.

<details>
<summary>Verify the exact dataset</summary>

SHA-256 of `dataset/23_Movies_and_TV_subset_50000.jsonl`:

```text
1b32d7588140db1d863a621dfec6bac35f727dd929e1a123965540c190f7fc73
```

From the repository root:

```bash
python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('dataset/23_Movies_and_TV_subset_50000.jsonl').read_bytes()).hexdigest())"
```

Keep the file's row order unchanged to reproduce the same random splits.

</details>

## Reproduce the experiments

### 1. Clone and install

Use **Python 3.12** and a virtual environment. Training uses scikit-learn on the CPU; no GPU or pretrained model download is required.

```bash
git clone https://github.com/wavepin/movie-review-helpfulness.git
cd movie-review-helpfulness
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --prefix .venv --name movie-review-helpfulness --display-name "Movie Review Helpfulness"
python -m jupyterlab
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it in PowerShell with `.venv\Scripts\Activate.ps1`, then run the same `python -m ...` commands.

### 2. Run the notebook

Open `output/jupyter-notebook/helpfulness-experiments-complete.ipynb`, select the **Movie Review Helpfulness** kernel, then choose **Restart Kernel and Run All Cells**. Start Jupyter from the repository root as above. The notebook finds the dataset by searching the working directory and its parents.

The notebook proceeds through:

| Sections | What happens |
| --- | --- |
| 1–4 | Setup, load data, prepare features, compare sample statistics |
| 5–7 | Explore vote imbalance, review year, length, rating, purchase status, and lexical style |
| 8–12 | Create splits, build pipelines, train seven models, compare validation performance |
| 13–15 | Tune the selected model's threshold, inspect test errors and text coefficients |
| 16–17 | Run exposed-only and mature-window experiments |
| 18 | Save a JSON summary of computed results |
| Method / Results | Read the preserved narrative of the original experiment |

A complete run produces inline tables and figures and writes `output/jupyter-notebook/helpfulness_final_results_summary.json`. That JSON is generated locally and excluded from Git. The closing Method and Results Markdown sections describe the original run and are **not automatically rewritten** when code is rerun. No trained model file is required or saved; models are fitted in notebook memory.

To execute without the Jupyter interface, from the repository root:

```bash
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=movie-review-helpfulness \
  --ExecutePreprocessor.timeout=1800 \
  --output helpfulness-experiments-rerun.ipynb \
  output/jupyter-notebook/helpfulness-experiments-complete.ipynb
```

This creates a separate executed notebook alongside the original. The timeout is per cell. If data discovery fails, verify that the `dataset/` directory is present and the notebook is running within this checkout. If imports fail, select the kernel installed from `.venv`.

### Repository layout

```text
movie-review-helpfulness/
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/
│   └── 23_Movies_and_TV_subset_50000.jsonl
└── output/
    └── jupyter-notebook/
        └── helpfulness-experiments-complete.ipynb
```

**Reproduction verified:** all 18 code cells completed on Python 3.12.14 with the pinned dependencies. The rerun reproduced the selected model, test F1 (0.579607), average precision (0.617491), and exact confusion matrix shown below.

The dependency pins define a reproduction environment; they are not a recovered lockfile from the original experiment, whose notebook metadata records Python 3.9.6. Library versions and numerical solvers can affect scores, coefficients, and the optimal threshold. The committed notebook preserves the original outputs used for the results below.

## Method

### Labels and splits

The main binary target is `any_helpful = int(helpful_vote >= 1)`. Two stratified `train_test_split` calls with `random_state=42` produce approximately 70% training, 15% validation, and 15% test data:

| Split | Reviews | Positive rate |
| --- | ---: | ---: |
| Training | 34,999 | ~25.3% |
| Validation | 7,501 | ~25.3% |
| Test | 7,500 | ~25.3% |

The one-row departure from an exact 35,000/7,500 split comes from the second split's fractional size and rounding. Vectorizers and numeric scaling are fitted on training data through scikit-learn pipelines.

### Text and engineered features

**Text:** title and body are joined, then represented with TF-IDF unigrams and bigrams, English stop-word removal, `min_df=2`, `sublinear_tf=True`, and a maximum of 20,000 features.

**Eleven engineered features:** rating, verified-purchase indicator, review year, word count, log word count, title word count, heuristic sentence count, exclamation count, question count, plot-term count, and emotion-term count. Numeric features are scaled using `StandardScaler(with_mean=False)` and combined with sparse TF-IDF features.

Small hand-built lexicons count plot terms such as *story*, *acting*, and *director*, and emotion terms such as *loved*, *boring*, and *amazing*. The notebook includes both complete lists. Plot/emotion groups and length quartiles support exploratory analysis; they are not additional classifier inputs. Length quartiles use ranked word counts with ties broken by row order.

### Models and evaluation

The comparison includes majority and stratified dummy baselines, Multinomial Naive Bayes (`alpha=0.5`), logistic regression, balanced logistic regression, balanced logistic regression with engineered features, and balanced linear SVM. Logistic regression uses `solver="liblinear"`, `max_iter=1000`, and default `C=1`; SVM uses `max_iter=5000`. Balanced models use `class_weight="balanced"`.

The main model is selected by **validation average precision**, with validation F1 as a tie-breaker. Here, “PR-AUC” refers to scikit-learn's `average_precision_score`, not trapezoidal integration of the precision–recall curve. The selected model's threshold is then tuned to maximize validation F1 and applied to test predictions. The model is not refitted on combined training and validation data.

The notebook also displays default-threshold test scores for all candidates, but its main selection code uses validation scores only. Accuracy, balanced accuracy, precision, recall, F1, and ROC-AUC provide additional context.

## Results

### Main model comparison

Original saved **validation** results, before threshold tuning:

| Model | Average precision | Precision | Recall | F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.253 | 0.000 | 0.000 | 0.000 | 0.500 |
| Stratified dummy | 0.251 | 0.245 | 0.241 | 0.243 | 0.495 |
| TF-IDF + Multinomial NB | 0.596 | 0.642 | 0.434 | 0.518 | 0.676 |
| TF-IDF + Logistic Regression | 0.605 | 0.684 | 0.382 | 0.490 | 0.661 |
| TF-IDF + Balanced Logistic Regression | 0.602 | 0.491 | 0.682 | 0.571 | 0.721 |
| **TF-IDF + Engineered Balanced Logistic Regression** | **0.622** | **0.511** | **0.698** | **0.590** | **0.736** |
| TF-IDF + Balanced Linear SVM | 0.542 | 0.471 | 0.590 | 0.524 | 0.683 |

### Selected model on the test set

At the validation-tuned threshold of approximately **0.525**:

| Average precision | F1 | Precision | Recall | Balanced accuracy | Accuracy | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.617 | 0.580 | 0.526 | 0.645 | 0.724 | 0.763 | 0.797 |

| Actual / Predicted | Zero votes | At least one vote |
| --- | ---: | ---: |
| Zero votes | 4,498 | 1,104 |
| At least one vote | 673 | 1,225 |

Class weighting improves recall relative to ordinary logistic regression, and the engineered model provides the strongest validation ranking performance. Threshold tuning changes precision and recall; it does not change average precision or ROC-AUC, which use the underlying scores.

### What the exploratory analysis shows

| Review group | Reviews | Observed-helpful rate |
| --- | ---: | ---: |
| Very short length quartile | 12,500 | 7.5% |
| Long length quartile | 12,500 | 56.6% |
| Both plot and emotion terms | 7,720 | 46.0% |
| Plot terms only | 7,251 | 30.9% |
| Emotion terms only | 12,652 | 22.9% |
| Neither lexical signal | 22,377 | 17.7% |

Longer reviews and reviews combining plot discussion with emotional language receive votes more often in this sample. These are associations: review length, age, vocabulary, and exposure can overlap, so the comparisons do not establish that a particular writing style causes more votes.

### Secondary experiments

| Experiment | Reviews | Positive rate | Reported model | Test F1 | Test average precision |
| --- | ---: | ---: | --- | ---: | ---: |
| Exposed-only: 2+ votes vs. exactly 1 | 12,654 | 51.8% | TF-IDF + Logistic Regression | 0.641 | 0.697 |
| Mature window: reviews through 2014, 1+ vs. 0 votes | 14,414 | 39.9% | TF-IDF + Engineered Balanced Logistic Regression | 0.684 | 0.762 |

These experiments use fresh stratified splits within each subset and default classifier thresholds. **The notebook chooses the displayed secondary “best” models by test average precision**, so these are exploratory best-test comparisons rather than independently selected final estimates. Different class prevalences and populations also prevent treating their higher scores as direct improvements over the main task. Older reviews have had longer to accumulate votes, but this restriction does not isolate exposure from other differences.

## Limitations and next steps

- **Exposure and label noise:** zero votes do not establish that a review is unhelpful; review year is only a proxy for opportunity to receive votes.
- **Evaluation scope:** splits are random at the review level, not grouped by product or reviewer and not chronological. Results do not establish performance on unseen products, reviewers, or future reviews. Exploratory analysis also covers the full sample.
- **Language representation:** TF-IDF and simple lexicons do not capture context, negation, discourse structure, or semantic equivalence. Linear coefficients describe associations rather than causes.
- **Uncertainty and generalization:** this is a single seeded split in one Amazon category, without confidence intervals or human helpfulness annotations.

Natural extensions include sentence embeddings or transformer models, richer entity and discourse features, explicit exposure modeling, grouped and temporal evaluation, cross-category testing, and human annotation. Improvements from these extensions would need to be measured.

## Credits and references

**Project team:** Jason Do, Bhavya Agarwal, and Khushi Sinha.

**Dataset:** McAuley Lab, [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/). Suggested dataset citation: Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024), *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952. The included reviews are third-party source data; this repository does not assign them a new license.

Related work informing the project:

- Kim, S.-M., Pantel, P., Chklovski, T., & Pennacchiotti, M. (2006). *Automatically Assessing Review Helpfulness*. EMNLP, 423–430.
- Mudambi, S. M., & Schuff, D. (2010). *What Makes a Helpful Online Review? A Study of Customer Reviews on Amazon.com*. MIS Quarterly, 34(1), 185–200.
- Saito, T., & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE, 10(3), e0118432.
