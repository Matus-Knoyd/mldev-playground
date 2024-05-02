ARG PYTHON_VERSION

FROM python:${PYTHON_VERSION}-slim

# Install required dependencies
RUN apt-get update \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /api
COPY . . 

RUN pip install mlflow fastapi uvicorn pandas
