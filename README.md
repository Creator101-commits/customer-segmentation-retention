# Customer Segmentation & Retention
> End-to-end churn modeling and retention planning pipeline for the Telco customer dataset.

[![Python Version][python-image]][python-url]
[![License: MIT][license-image]][license-url]
[![Repo][repo-image]][repo-url]

This project builds a full customer retention workflow using feature engineering, calibrated probabilities, threshold optimization, and business cost analysis. It produces model metrics, one-page summaries, recommendations, and charts that are directly usable for campaign planning.

The current production recommendation uses a recall-constrained threshold to balance retention coverage with intervention efficiency.

## Installation

OS X & Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib scikit-learn xgboost shap optuna lightgbm
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib scikit-learn xgboost shap optuna lightgbm
```

## Usage example

Run the final optimization pipeline:

```sh
python model_search_ensemble.py
```

Generate charts from the latest saved report:

```sh
python generate_result_graphs.py
```

Open and run the notebook (optional interactive flow):

```sh
code telco_churn_runbook.ipynb
```

Key outputs:
- `ensemble_optimization_report.json`
- `recommendation.txt`
- `RESULTS_ONE_PAGE.md`
- `Results/results_metrics_comparison.png`
- `Results/results_model_summary.png`
- `Results/results_business_cost_breakdown.png`

## Development setup

Run the complete local workflow:

```sh
source .venv/bin/activate
python PythonCode/model_search_ensemble.py
python PythonCode/generate_result_graphs.py
```

Optional quick verification:

```sh
python -m py_compile model_search_ensemble.py generate_result_graphs.py
```

## Release History

* 1.0.0
    * ADD: Final calibrated XGBoost + LightGBM comparison pipeline
    * ADD: Business-cost and recall-constrained threshold reporting
    * ADD: One-page result summary and recommendation deliverables
* 0.9.0
    * ADD: Feature engineering, calibration, and Optuna tuning workflow
    * ADD: Result chart generation scripts
* 0.1.0
    * Initial project scaffold and baseline churn modeling

## Meta

Maintainer: Creator101-commits

Distributed under the MIT license. See `LICENSE` for more information.

[https://github.com/Creator101-commits/customer-segmentation-retention](https://github.com/Creator101-commits/customer-segmentation-retention)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-change`)
3. Commit your changes (`git commit -m "Describe your change"`)
4. Push to the branch (`git push origin feature/your-change`)
5. Open a Pull Request

<!-- Markdown link & img dfn's -->
[python-image]: https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square
[python-url]: https://www.python.org/
[license-image]: https://img.shields.io/badge/license-MIT-green?style=flat-square
[license-url]: https://opensource.org/licenses/MIT
[repo-image]: https://img.shields.io/badge/repo-GitHub-black?style=flat-square
[repo-url]: https://github.com/Creator101-commits/customer-segmentation-retention
