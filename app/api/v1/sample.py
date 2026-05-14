from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pdt.common.logger import get_logger

from app.api.v1.dto.sample_dto import SampleDTO
from app.services.sample_service import SampleService


@dataclass
class UserInfo:
    """Holds user information extracted from request.state."""

    gy_user_id: str = "unknown"
    gy_user_mail: str = "unknown"
    user_roles: set[str] = field(default_factory=set)
    user_groups: set[str] = field(default_factory=set)


sample_router = APIRouter()


@sample_router.post("/sample-api")
async def sample_api(payload: SampleDTO):
    try:
        sample_handler = SampleService()
        result = sample_handler.process_request(payload=payload)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "message": "Successfully fetched details",
                "data": result,
            },
        )
    except Exception as e:
        get_logger().error(f"Error in sample_api: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Internal server error"},
        )


def get_user_info(args: tuple, kwargs: dict[str, Any]) -> UserInfo:
    """
    Extract user information from request.state.

    Returns a UserInfo with safe defaults if request is not found.
    """
    request = kwargs.get("request")

    if request is None:
        for arg in args:
            if hasattr(arg, "state"):
                request = arg
                break

    if request is None or not hasattr(request, "state"):
        return UserInfo()

    return UserInfo(
        gy_user_id=getattr(request.state, "gy_user_id", "unknown") or "unknown",
        gy_user_mail=getattr(request.state, "gy_user_mail", "unknown") or "unknown",
        user_roles=getattr(request.state, "user_roles", set()) or set(),
        user_groups=getattr(request.state, "user_groups", set()) or set(),
    )
