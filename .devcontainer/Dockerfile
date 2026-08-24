# Dev container for IAG-2024B-GenerativeQA on an old Ubuntu 18.04 host.
#
# The container uses Ubuntu 24.04 + CUDA 11.8 libraries so the conda env
# can install PyTorch cu118 wheels compatible with the old host's GPU driver.
# The actual GPU driver comes from the host via NVIDIA Container Toolkit
# (devcontainer.json passes --gpus=all).

FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# NVIDIA CUDA 11.8 userland libraries
# ---------------------------------------------------------------------------
# CUDA 11.8 predates Ubuntu 24.04, so we install from NVIDIA's ubuntu2204 repo.
# The host driver (libcuda.so / nvidia-smi) is injected at container runtime.
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-libraries-11-8 \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb

# ---------------------------------------------------------------------------
# Miniconda
# ---------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" config --add channels conda-forge \
    && "${CONDA_DIR}/bin/conda" config --set channel_priority strict \
    && "${CONDA_DIR}/bin/conda" config --set env_prompt '({name})'

# Accept Anaconda ToS for the defaults channel used by the env file (conda >=24).
RUN if "${CONDA_DIR}/bin/conda" tos --help >/dev/null 2>&1; then \
        "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
        "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r; \
    fi

# ---------------------------------------------------------------------------
# Bake the conda environment into the image
# ---------------------------------------------------------------------------
COPY .devcontainer/environment.cuda118.yml /tmp/environment.cuda118.yml
RUN "${CONDA_DIR}/bin/conda" env create -f /tmp/environment.cuda118.yml \
    && "${CONDA_DIR}/bin/conda" clean -afy \
    && rm -f /tmp/environment.cuda118.yml

# ---------------------------------------------------------------------------
# Permissions and defaults for the vscode user
# ---------------------------------------------------------------------------
RUN groupadd -r conda \
    && usermod -aG conda vscode \
    && chown -R vscode:conda "${CONDA_DIR}" \
    && chmod -R g+r+w "${CONDA_DIR}" \
    && find "${CONDA_DIR}" -type d -print0 | xargs -0 -n1 chmod g+s

# Make the baked environment the default for shells and VS Code.
ENV PATH="${CONDA_DIR}/envs/pyt-eqa-fge/bin:${PATH}"
RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate pyt-eqa-fge" >> /home/vscode/.bashrc
