#!/usr/bin/env python3
"""Build decision-support outputs for Lesson 16.

This script translates model communication outputs into decision-support artifacts.
It is intentionally simple and transparent so learners can inspect the logic.
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    communication_summary_path = reports_dir / "diabetes-communication-summary.md"

    if communication_summary_path.exists():
        communication_summary_status = "Found communication summary from Lesson 15."
    else:
        communication_summary_status = (
            "Communication summary from Lesson 15 was not found. "
            "This decision summary still runs, but Lesson 15 should be run first "
            "for the complete workflow."
        )

    framing_df = pd.DataFrame(
        {
            "decision_component": [
                "Goal",
                "Model evidence",
                "Possible use",
                "Main uncertainty",
                "Reasonable action",
                "Unsupported action",
            ],
            "decision_framing": [
                "Identify individuals who may have higher predicted disease progression.",
                "The model shows moderate predictive ability and uses features such as BMI in prediction.",
                "Use predictions as one input for prioritizing closer monitoring or further review.",
                "Predictions have error and are based on associations in this dataset.",
                "Flag higher-risk cases for review while combining model output with domain expertise.",
                "Do not use the model alone to prescribe treatment or claim causation.",
            ],
        }
    )

    framing_path = reports_dir / "diabetes-decision-framing-table.csv"
    framing_df.to_csv(framing_path, index=False)

    decision_summary = f"""# Diabetes Model Decision Summary

## Workflow Status
{communication_summary_status}

## Objective
Identify individuals who may have higher predicted disease progression using clinical features.

## Analytical Finding
The model shows moderate predictive ability and reasonably stable performance across evaluation splits. Features such as BMI are used by the model when estimating disease progression.

## Interpretation
BMI and related features are useful predictors in this dataset, but the model describes associations and prediction patterns rather than causal mechanisms.

## Decision Context
Predictions may support prioritization for closer monitoring, further review, or additional data collection. The model should not be used as the sole basis for treatment decisions.

## Key Limitation
The model has prediction error and was trained on a specific dataset. Decisions should account for uncertainty, clinical context, and consequences of false positives or false negatives.

## Recommended Use
Use the model as a decision-support tool, not as an automatic decision-maker.
"""

    summary_path = reports_dir / "diabetes-decision-summary.md"
    summary_path.write_text(decision_summary)

    print("Decision-support outputs created:")
    print(f"- {framing_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
