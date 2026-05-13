# set base image (host OS)
FROM python:3.11.6-bookworm AS builder

ARG INSTALL_DEV=false
ARG GITHUB_USER="x-access-token"
ARG GITHUB_PASS=$GITHUB_PASS
ARG NEXUS_USER=$NEXUS_USER
ARG NEXUS_PASS=$NEXUS_PASS

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.8.2 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

# install poetry - respects $POETRY_VERSION and $POETRY_HOME
RUN python -m pip install poetry==${POETRY_VERSION} \
    && poetry --version

WORKDIR /project

COPY poetry.lock pyproject.toml ./

RUN echo "machine github.com login $GITHUB_USER password $GITHUB_PASS" >> ~/.netrc \
    && poetry config http-basic.nexus $NEXUS_USER $NEXUS_PASS \
    && poetry install --no-root $(test "$INSTALL_DEV" == false && echo "--without dev") \
    && rm -f ~/.netrc \
    && poetry config http-basic.nexus --unset

FROM python:3.11.6-slim AS runtime

ENV VIRTUAL_ENV=/project/.venv \
    PATH="/project/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /project

COPY ./app ./app
COPY ./tests ./tests

CMD [ "python", "-m", "app.main" ]
