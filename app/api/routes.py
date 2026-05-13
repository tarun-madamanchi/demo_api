from fastapi import APIRouter

from app.api.checks import check_router
from app.api.v1.sample import sample_router

router = APIRouter()

router.include_router(check_router, tags=["Check Api's"])
router.include_router(sample_router, tags=["sample Api's"], prefix="/sample")
