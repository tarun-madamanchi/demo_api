from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pdt.common.logger import get_logger

from app.api.v1.dto.sample_dto import SampleDTO
from app.services.sample_service import SampleService

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
