# Value at Risk estimation using Natural Gradient Boosting for Probabilistic Prediction modeling
Author: Michał Woźniak (id 385190)

## Project structure description
The `dataset` directory contains:
* spx.csv - raw data used in the project
* spx_ngboost_final_dataset.parquet - preprocessed data (in the notebook: experiments/5_ngboost_fe_and_fs.ipynb) used in the project

The `docs` directory contains:
* ML2_project_proposal_385190.pdf - project proposal
* ML2_report_385190.pdf - project report
* ML2_presentation_385190.pdf - project presentation

The `experiments` directory contains:
* 1_exploratory_data_analysis.ipynb - covers statistical exploratory data analysis of the dataset - S&P 500 stock returns data (one-dimensional analysis)
* 2_simple_statistical_models.ipynb - covers simple statistical modeling and final backtesting of the Value at Risk
* 3_econometric_models.ipynb - covers econometric modeling and final backtesting of the Value at Risk
* 4_ngboost_eda.ipynb - covers exploratory data analysis of the whole dataset (multi-dimensional analysis) for NGBoost modeling
* 5_ngboost_fe_and_fs.ipynb - covers features engineering and features selection in the NGBoost environment 
* 6_ngboost_functional_form_choice_and_hyperparameters_tuning.ipynb - covers functional form choice and hyperparameters tuning in the NGBoost environment 
* 7_ngboost_first_VaR_approach.ipynb - covers first NGboost approach to the modeling (VaR as a quantile of the predicted distribution) and final backtesting of the Value at Risk
* 8_ngboost_second_VaR_approach.ipynb - covers second NGboost approach to the modeling (VaR as a quasi GARCH approach) and final backtesting of the Value at Risk
* 9_ngboost_ensembling_with_GARCH.ipynb - covers ensembling approach to the modeling (simple mean of the NGboost first approach and QML-GARCH_1_1) and final backtesting of the Value at Risk
* 10_ngboost_switching_with_GARCH.ipynb - covers switching approach to the modeling (switching between the NGboost first approach and QML-GARCH_1_1) and final backtesting of the Value at Risk
* mp_cv.py - set of modeling functions which allowed for the multiprocessing (they were used in files with following prefixes: 6,7,8,9,10)
* var_test.py - set of Value at Risk formal statistical tests (source: https://github.com/dkaszynski/VVaR)
