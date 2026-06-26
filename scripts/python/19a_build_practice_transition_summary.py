#!/usr/bin/env python3
"""
Build final practice-transition artifacts for Lesson 19.

This script creates a small roadmap table and a Markdown summary that connect
Applied Data Science foundations to real-world analytical systems.
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    roadmap = pd.DataFrame(
        {
            "stage": [
                "Data structure",
                "Modeling",
                "Evaluation",
                "Interpretation",
                "Communication",
                "Decision context",
                "Responsible use",
                "System transition",
            ],
            "core_question": [
                "Is the data organized enough to support analysis?",
                "Can the data support useful prediction or explanation?",
                "How stable and reliable is the model?",
                "What is the model using to make predictions?",
                "Can the results be explained clearly?",
                "How could the results inform action?",
                "What are the limits and risks?",
                "How would this become a maintained workflow?",
            ],
            "real_world_output": [
                "Documented dataset and feature table",
                "Saved model and baseline results",
                "Metrics, plots, and validation summaries",
                "Feature influence tables and interpretation notes",
                "Clear written summary",
                "Decision framing table",
                "Limitations and responsible use checklist",
                "Deployment and monitoring roadmap",
            ],
        }
    )

    roadmap_path = reports_dir / "applied-data-science-practice-roadmap.csv"
    roadmap.to_csv(roadmap_path, index=False)

    summary = """# Applied Data Science Practice Transition

The Applied Data Science System provides a foundation for moving from isolated analysis to real-world analytical systems.

## Core transition

The workflow moves from:

```text
data → model → metric
```

toward:

```text
data → reproducible workflow → evaluation → interpretation → communication → decision context → responsible use
```

## What this enables

A completed applied data science workflow should make it possible to answer:

- What data was used?
- What code produced the result?
- How was the result evaluated?
- What does the result mean?
- What does it not mean?
- How could the result support a decision?
- What are the limitations and risks?

## Next system layer

The next CDI layer is ML → Deployment → DevOps, where analytical workflows become maintained systems.

That layer adds:

- model packaging
- prediction APIs
- deployment
- monitoring
- drift detection
- retraining workflows
- operational documentation

## Closing statement

Strong systems require strong analytical reasoning. Deployment should extend a defensible workflow, not hide weak assumptions.
"""

    summary_path = reports_dir / "applied-data-science-practice-transition.md"
    summary_path.write_text(summary, encoding="utf-8")

    print("Created:")
    print(f"- {roadmap_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
