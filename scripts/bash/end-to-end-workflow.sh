#!/usr/bin/env bash

set -euo pipefail

echo "========================================"
echo "Applied Data Science System"
echo "End-to-End Workflow"
echo "========================================"

echo
echo "Checking Python environment..."
python scripts/python/01b_check_environment.py

echo
echo "Creating example raw dataset..."
python scripts/python/08a_create_example_people_dataset.py

echo
echo "Building model-ready feature table..."
python scripts/python/08b_build_feature_table.py

echo
echo "Saving diabetes dataset..."
python scripts/python/09a_save_diabetes_dataset.py

echo
echo "Training baseline linear model..."
python scripts/python/09b_train_baseline_linear_model.py

echo
echo "Plotting observed vs predicted values..."
python scripts/python/09c_plot_observed_vs_predicted.py

echo
echo "Evaluating repeated train/test splits..."
python scripts/python/10a_evaluate_repeated_splits.py

echo
echo "Running cross-validation..."
python scripts/python/10b_cross_validate_linear_model.py

echo
echo "Plotting residuals..."
python scripts/python/10c_plot_residuals.py

echo
echo "Comparing model variants..."
python scripts/python/11a_compare_model_variants.py

echo
echo "Running pipeline cross-validation..."
python scripts/python/12a_cross_validate_pipeline.py

echo
echo "Interpreting model features..."
python scripts/python/13a_interpret_model_features.py

echo
echo "Translating model outputs to claims..."
python scripts/python/14a_translate_model_outputs_to_claims.py

echo
echo "Building results communication summary..."
python scripts/python/15a_build_results_communication_summary.py

echo
echo "Building decision summary..."
python scripts/python/16a_build_decision_summary.py

echo
echo "Running end-to-end case study..."
python scripts/python/17a_run_end_to_end_case_study.py

echo
echo "Building responsible use review..."
python scripts/python/18a_build_responsible_use_review.py

echo
echo "Building practice transition summary..."
python scripts/python/19a_build_practice_transition_summary.py

echo
echo "Rendering Quarto site..."
quarto render

echo
echo "========================================"
echo "End-to-end workflow complete."
echo "Rendered site: docs/index.html"
echo "========================================"