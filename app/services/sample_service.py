from __future__ import annotations

from typing import Any, Dict

from pdt.common.logger import get_logger

from app.api.v1.dto.sample_dto import SampleDTO
from app.config import Config
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserInfo:
    """Holds user information extracted from request.state."""

    gy_user_id: str = "unknown"
    gy_user_mail: str = "unknown"
    user_roles: set[str] = field(default_factory=set)
    user_groups: set[str] = field(default_factory=set)


class SampleService:
    """
    Sample handler demonstrating clean microservice architecture.

    This class serves as a template for implementing business logic services
    in the PDT FastAPI microservice framework. It follows the separation of
    concerns principle by keeping request handling, validation, and business
    logic isolated from the API layer.

    Responsibilities:
    - Process incoming requests from API endpoints
    - Validate and sanitize input data
    - Execute core business logic
    - Return structured responses
    - Handle exceptions gracefully

    Usage Example:
        # >>> handler = SampleHandler()
        # >>> response = handler.process_request({"user_id": 123})
        # >>> print(response)
        {"status": "success", "data": {...}}

    Note:
        This is a demo implementation. In production, inject dependencies
        via constructor (dependency injection) rather than importing globally.
    """

    def __init__(self):
        self._config = Config

    def process_request(self, payload: SampleDTO) -> Dict[str, Any]:
        """
        Process a sample request and return structured response.

        Args:
            request_data SampleSchema: Input data from API endpoint

        Returns:
            Dict[str, Any]: Standardized response with status and data

        Raises:
            ValueError: If required fields are missing
            Exception: For any unexpected errors (logged and wrapped)
        """
        try:
            get_logger().info(
                "Processing sample request",
                extra={"payload": payload.model_dump()},
            )

            # Simulate business logic

            result = {
                "app_id": self._config.APP_BUILD_ID,
                "processed_by": "SampleHandler",
                "user_id": payload.user_id,
                "environment": self._config.ENVIRONMENT,
            }

            get_logger().debug(
                "Request processed successfully", extra={"result": result}
            )

            return result

        except Exception as e:
            get_logger().error(str(e))
            raise
