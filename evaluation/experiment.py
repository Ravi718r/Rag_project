import uuid
import json
from pathlib import Path
from datetime import datetime

REPORT_DIR = Path("evaluation/reports")

def create_experiment(config=None):
    """
    Create metadata for one evaluation experiment.
    """

    experiment = {
        "experiment_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "config": config or {}
    }

    return experiment

def save_experiment(experiment, summary):
    """
    Save experiment metadata and summary.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    experiment_data = {
        **experiment,
        "summary": summary
    }

    path = (
        REPORT_DIR
        / f"{experiment['experiment_id']}.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            experiment_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    return path


