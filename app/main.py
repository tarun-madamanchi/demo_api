import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from pdt.common.logger import get_logger

from app.api.routes import router
from app.config import Config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Context manager to manage the lifespan of the FastAPI application.

    At start time, this function sets up the connections and starts it.
    It also sets up the monitoring for the FastAPI application.

    At stop time, this function stops the connections and tears down
    the monitoring for the FastAPI application.

    Args:
        app: The FastAPI application instance.

    Returns:
        The context manager for the entire FastAPI application.
    """
    yield


def create_app():
    """Factory to create FastAPI app instance."""
    app = FastAPI(root_path=Config.ROOT_PATH, lifespan=lifespan)

    app.include_router(router)

    return app


app = create_app()

if __name__ == "__main__":
    get_logger().info("Starting ASGI server")
    base = os.path.dirname(__file__)
    log_conf_file = os.path.join(base, "./uvicorn/log_conf.yml")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.APP_PORT,
        lifespan="on",
        log_config=log_conf_file,
    )
