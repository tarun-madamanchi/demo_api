# pdt-template-fastapi
A template repository for building FastAPI applications at Goodyear PDT, with clean project structure and logging, and Docker support.


## 🚀 Features
- FastAPI-based template
- Clean folder structure
- Built-in logging configuration
- Docker-ready
- Poetry for dependency management
- Pre-commit hooks enabled

---

## 📁 Template Structure

```pdt-template-fastapi/
│
├── app/ # Application source code
│ ├── config.py # API configuration
│ │
│ ├── api/ # API route definitions
│ │ ├── v1/ # Version 1 of the API's
│ │ │ ├── dto/ # API DTO Pydantic Models
│ │ │ │ ├── init.py
│ │ │ │ ├── sample_dto.py
│ │ │ ├── init.py
│ │ │ └── sample.py
│ │ ├── init.py
│ │ ├── checks.py # Health check endpoints
│ │ ├── routes.py # API route registrations
│ │
│ ├── services/ # Business logic
│ │ ├── init.py
│ │ └── sample_service.py
│ │
│ ├── uvicorn/ # Uvicorn server configuration
│ │ └── log_conf.yml
│ │
│ ├── init.py
│ ├── main.py # FastAPI application entrypoint with lifecycle
│
├── tests/ # Unit tests
│
├── Dockerfile
├── pyproject.toml
├── poetry.lock
├── .dockerignore
└── README.md
```

---

## 🛠️ Local Development Setup

## Local Usage
```
poetry install
export APP_PORT=8080 # or any other port as you cannot bind to default port 80 without root permissions
poetry run start
```

## Docker Usage

```bash
export GITHUB_PASS=<your_personal_access_token>

docker build \
    --build-arg GITHUB_PASS=${GITHUB_PASS} \
    -t your_app_name .

docker run -p 8080:80 your_app_name
```

This will run the container locally on port 8080, by binding the container port 80 to the host port 8080.

You can establish and store connections to other databases or services in the same way.

# Pre-Commit
Pre-commit hooks help catch simple issues before submitting code for review. Hooks are executed automatically on git commit. If a hook fails, the commit is blocked, and you must fix the issues before retrying.
## Install Pre-Commit
```
pip install pre-commit
```

## Initialize Pre-Commit
```
pre-commit install
```

## Run Hooks Manually

Run all hooks on all files (useful before pushing):
```
pre-commit run --all-files
```
