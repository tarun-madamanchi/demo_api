import os


class Config(object):
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "LOCAL")

    HOSTNAME = os.getenv("HOSTNAME", "")
    BASE_HOSTNAME = os.getenv("BASE_HOSTNAME", "")

    IS_TESTING = ENVIRONMENT == "TESTING"
    IS_LOCAL = ENVIRONMENT == "LOCAL"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    ROOT_PATH = "/api" if ENVIRONMENT != "LOCAL" else ""

    APP_PORT = int(os.getenv("APP_PORT", 80))

    APP_BUILD_ID = os.getenv("BUILD_ID")
    APP_BUILD_TAG = os.getenv("BUILD_TAG")

    AWS_PROFILE = os.getenv("AWS_PROFILE", "default")

    AWS_ATHENA_ROLE_ARN = os.getenv("AWS_ATHENA_ROLE_ARN", None)
    AWS_ATHENA_WORKGROUP = os.getenv("AWS_ATHENA_WORKGROUP", None)
    AWS_ATHENA_S3_STAGING_DIR = os.getenv("AWS_ATHENA_S3_STAGING_DIR", None)
