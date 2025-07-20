# nasrine_seraji

This project is part of the Bain's recruitment process. This project is a transform ation of a Jupyter
notebook-based property valuation model into a scalable, containerized service. 
The end result should be a production-ready machine learning API for predicting Chilean property prices. 

## Installation and Quick Start

### Prerequisites

- **Docker Desktop**: Install from the [official Docker documentation](https://docs.docker.com/get-docker/)
- **Git**: For cloning the repository

### Demo: Running the API

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd nasrine_seraji
   ```

2. **Train the model** (first time setup):
   ```bash
   docker compose -f docker/docker-compose.dev.yml run --rm --build property_friends
   cd property_friends && uv run scripts/train_and_serialize_model.py
   ```

3. **Start the API service**:
   ```bash
   docker compose -f docker/docker-compose.yml up --build property-friends-api
   ```

4. **Test the API**:
   - **Interactive docs**: Visit [http://localhost:8000/docs](http://localhost:8000/docs)
   - **API key**: Located in `api/.env` (default: `your-secret-api-key-here`)
   
   **Example cURL request**:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/predictions/' \
     -H 'accept: application/json' \
     -H 'Authorization: Bearer your-secret-api-key-here' \
     -H 'Content-Type: application/json' \
     -d '{
       "type": "departamento",
       "sector": "vitacura",
       "net_usable_area": 140.0,
       "net_area": 170.0,
       "n_rooms": 4,
       "n_bathroom": 4,
       "latitude": -33.40,
       "longitude": -70.58
     }'
   ```

## Assumptions
 
 Here is a brief summary of the assumptions I took when starting the devlopment.
- **Service Separation**: API and training run as separate services. The API provides inference only - training is managed independently by the data science team.
- **Model Storage**: Currently uses filesystem-based model storage. In production, this would be cloud blob storage with versioning (S3, Azure Blob, etc.), or proper
model management system like mlflow or weight and biases
- **Deployment**: As requested, the project is designed for containerized deployment with Docker and docker-compose for local development and testing. Here I am
assuming that `Docker Desktop` is installed on the devs computer.

## Architecture

The project is structured into two main components:

### Core ML Package (`property_friends/`)
- **Models**: Training, prediction, and preprocessing utilities
- **Data Input**: CSV file loading and data handling
- **Scripts**: Command-line tools for model training and serialization
- **Testing**: Comprehensive test suite

### API Service (`api/`)
- **FastAPI Application**: RESTful endpoints for property price predictions
- **Authentication**: Bearer token security with configurable API keys
- **Request/Response Models**: Pydantic schemas for data validation
- **Docker Integration**: Containerized deployment with health checks


### Training Pipeline (`training_pipeline/`)
- **Dagster**: Production-ready pipeline manager that demonstrates enterprise-grade ML orchestration capabilities while being fully functional for real production use
- **Asset-Based Architecture**: Each training step (data loading, preprocessing, model training, evaluation, serialization) is defined as a Dagster asset with clear dependencies
- **Web Interface**: Visual pipeline monitoring and execution through Dagster's web UI at `localhost:3000`
- **Role**: Orchestrates calls to the core ML package with proper dependency management and observability

## Technology Stack

- **Docker**: Ensures consistent deployment across environments and simplifies dependency management. Containerization provides isolation and reproducibility.
- **uv**: Ultra-fast Python package manager that significantly speeds up dependency resolution and installation compared to pip. Provides reliable dependency locking.
- **FastAPI**: Modern, fast web framework with automatic API documentation and built-in data validation. Excellent performance and developer experience with async support.
- **Pydantic**: Robust data validation using Python type annotations. Ensures request/response data integrity and provides clear error messages.
- **pytest**: Comprehensive testing framework with fixtures and powerful assertion capabilities. Essential for maintaining code quality in production systems.
- **Dagster**: Modern data orchestration platform with asset-based architecture and rich web UI. Provides reliable pipeline execution, dependency tracking, and observability for ML workflows.


## Development Guide

### Setting Up Development Environment

1. **Install pre-commit hooks**:
   ```bash
   pip install --user pipx
   pipx ensurepath
   pipx install pre-commit
   pre-commit install
   pre-commit run --all-files
   ```

### Running Tests

2. **Run tests locally**:
   ```bash
   docker compose -f docker/docker-compose.dev.yml run --rm --build property_friends
   cd property_friends && uv run pytest tests
   ```

### Model Training

3. **Train model locally with the pipeline**:
   ```bash
   docker compose -f docker/docker-compose.yml up --build
   # Go to localhost:3000, "assets", select all assests, materialize the assets
   ```

3. **Train model locally with a script (deprecated)**:
   ```bash
   docker compose -f docker/docker-compose.dev.yml run --rm --build property_friends
   cd property_friends && uv run scripts/train_and_serialize_model.py
   ```

### Release Management

#### Releasing a New Version

To release a new version of the property_friends package:

1. **Update the version number**:
   ```bash
   # Edit property_friends/property_friends/__init__.py
   # Change __version__ = "0.2.0" (example)
   cd property_friends && uv build
   ```

2. **Update API dependency**:
   ```bash
   # Edit api/pyproject.toml
   # Update wheel path: property_friends-NEW_VERSION-py3-none-any.whl
   cd api && uv lock
   ```

3. **Build and test**:
   ```bash
   docker compose -f docker/docker-compose.api.yml up --build
   ```

## Future Improvements

### Production Deployment
- **Cloud Infrastructure**: Deploy to cloud providers (AWS, GCP, Azure) with proper load balancing and auto-scaling
- **Container Registry**: Publish Docker images to registries (Docker Hub, ECR) for faster CI/CD pipelines
- **Security**: Implement proper secret management and token handling for cloud environments

### Development Experience
- **Package Publishing**: Publish `property_friends` as a proper Python package for cleaner dependency management
- **Automation**: Create streamlined scripts/Makefile for common development tasks (test, build, deploy)
- **Repository Management**: Implement branch protection rules and advanced GitHub workflows

### Model Operations (MLOps)
- **Enhanced Dagster Integration**: Leverage Dagster's asset versioning and materialization system to automatically version models and serve them directly to the API, eliminating redundant filesystem serialization
- **Model Versioning**: Integrate with MLflow or Weights & Biases for proper model lifecycle management
- **Monitoring**: Add model performance monitoring and data drift detection
- **System Testing**: Implement end-to-end API tests with realistic property data scenarios

### Scalability & Reliability
- **Database Integration**: Replace filesystem storage with proper database for model artifacts
- **Observability**: Add comprehensive logging, metrics, and distributed tracing

## Development Log

For transparency and my own learning curiosity, I kept a rough log of development work. It's interesting to track how long tasks actually take versus initial estimates:
- 0h45m : explo notebook, basic tech choices
- 2h15m : Base of project: docker, payload project, test, CI
- 3h05m : Preprocessor, format and lint
- 4h00m : property_friends package implemented, mypy, refacto
- 4h15m : fix CI follow up refacto
- 4h45m : local train with script (ugly but needed for time constraint), work in parallel on API
- 5h20m : API done and connected to main payload
- 5h40m : Rewrite README
- 6h00m : Refacto data loading
- 7h30m : Pipeline training with Dagster
