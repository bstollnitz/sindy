# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

# Install dependencies of OpenCV that don't come with it.
RUN apt-get update
RUN apt-get install libgl1 -y

# Update the conda environment according to the conda.yml file in the project.
COPY conda.yml /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/conda.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/conda.yml; fi \
    && rm -rf /tmp/conda-tmp
