#!/usr/bin/env python3
"""
Build responsible-use outputs for Lesson 18.

This script creates reusable project artifacts describing model limitations,
responsible-use boundaries, and practical safeguards for the diabetes modeling workflow.
"""

from pathlib import Path

import pandas as pd


REPORTS_DIR = Path("reports")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    responsible_use_checklist = pd.DataFrame(
        {
            "question": [
                "Was the dataset source and scope described?",
                "Were model assumptions and limitations stated?",
                "Was performance evaluated beyond a single split?",
                "Were feature interpretations framed as model behavior?",
                "Were causal claims avoided?",
                "Was uncertainty communicated?",
                "Was the intended use clearly bounded?",
                "Were inappropriate uses explicitly discouraged?",
            ],
            "status": [
                "yes",
                "yes",
                "yes",
                "yes",
                "yes",
                "yes",
                "yes",
                "yes",
            ],
            "evidence": [
                "Dataset loaded from data/diabetes.csv and described as a structured teaching dataset.",
                "Linear model assumptions and model simplification were discussed.",
                "Cross-validation summaries were used to assess stability.",
                "Coefficients were interpreted as model behavior, not biological truth.",
                "Association and causation were explicitly separated.",
                "Prediction error and variability across folds were discussed.",
                "Use was framed as screening, prioritization, or monitoring support.",
                "Standalone clinical decision-making was discouraged.",
            ],
        }
    )

    limitations_table = pd.DataFrame(
        {
            "limitation_area": [
                "Dataset scope",
                "Feature coverage",
                "Model structure",
                "Generalization",
                "Causality",
                "Individual prediction error",
                "Decision use",
            ],
            "limitation": [
                "The dataset represents a specific sample and should not be assumed to represent all populations.",
                "Only available measured variables can be used by the model.",
                "Linear regression simplifies relationships as additive and linear.",
                "Cross-validation estimates internal stability but does not replace external validation.",
                "Predictive associations do not establish causal mechanisms.",
                "Individual predictions may have meaningful error even when average performance is acceptable.",
                "The model should support judgment, not replace domain expertise or clinical decision-making.",
            ],
            "responsible_response": [
                "Describe the data source and avoid broad claims beyond the dataset.",
                "State that unmeasured factors may influence outcomes.",
                "Treat the model as an interpretable approximation.",
                "Validate on new data before deployment in a new setting.",
                "Use association language unless causal evidence is available.",
                "Communicate prediction uncertainty and avoid exact individual claims.",
                "Use outputs as decision support only within a defined context.",
            ],
        }
    )

    summary_text = """# Responsible Use Summary

This model provides moderate predictive support within the diabetes dataset.
Its outputs are useful for understanding model behavior and demonstrating a reproducible applied data science workflow.

However, the model should not be interpreted causally, should not be assumed to generalize to all populations, and should not be used as a standalone clinical decision tool.

Any real-world use would require additional validation, domain review, and careful consideration of prediction errors and consequences.
"""

    checklist_path = REPORTS_DIR / "diabetes-responsible-use-checklist.csv"
    limitations_path = REPORTS_DIR / "diabetes-limitations-table.csv"
    summary_path = REPORTS_DIR / "diabetes-responsible-use-summary.md"

    responsible_use_checklist.to_csv(checklist_path, index=False)
    limitations_table.to_csv(limitations_path, index=False)
    summary_path.write_text(summary_text)

    print("Responsible-use outputs written:")
    print(f"- {checklist_path}")
    print(f"- {limitations_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
