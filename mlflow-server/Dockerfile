ARG PYTHON_VERSION

FROM python:${PYTHON_VERSION}-slim

# Install required dependencies
RUN apt-get update \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

RUN pip install mlflow








