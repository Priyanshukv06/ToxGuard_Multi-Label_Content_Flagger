"""
Data router — serves random test data samples.
"""

import os
import json
import random
import logging
from fastapi import APIRouter
from app.models import RandomCommentResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/data", tags=["Sample Data"])

_sample_data: list[dict] = []


def load_sample_data():
    global _sample_data

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_file = os.path.join(project_root, "data_sample", "test_sample.json")

    if os.path.exists(sample_file):
        with open(sample_file, "r") as f:
            _sample_data = json.load(f)
        logger.info(f"✅ Loaded {len(_sample_data)} sample records")
    else:
        logger.warning(f"⚠️ Sample data not found: {sample_file}")


@router.get("/random", response_model=RandomCommentResponse)
async def get_random_comment():
    if not _sample_data:
        return RandomCommentResponse(comment={"error": "No sample data loaded."})
    return RandomCommentResponse(comment=random.choice(_sample_data))


@router.get("/random/batch")
async def get_random_comments(count: int = 5):
    if not _sample_data:
        return {"comments": [], "error": "No sample data loaded."}

    count = min(count, 50)
    comments = random.sample(_sample_data, min(count, len(_sample_data)))
    return {"comments": comments}
