FROM ubuntu:22.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy files
COPY pyproject.toml /home/ubuntu/immuneML/
COPY requirements.txt /home/ubuntu/immuneML/
COPY requirements-docker.txt /home/ubuntu/immuneML/
COPY requirements_TCRdist.txt /home/ubuntu/immuneML/
COPY requirements_DeepRC.txt /home/ubuntu/immuneML/
COPY README.md /home/ubuntu/immuneML/
COPY immuneML /home/ubuntu/immuneML/immuneML
COPY test /home/ubuntu/immuneML/test
COPY scripts /home/ubuntu/immuneML/scripts

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git-all gcc g++ gfortran make \
    python3.11 python3.11-dev parasail \
    autotools-dev autoconf libtool pkgconf && \
    rm -rf /var/lib/apt/lists/*

# Install the dependency CompAIRR
RUN git clone https://github.com/uio-bmi/compairr.git /home/ubuntu/compairr_folder && \
    make -C /home/ubuntu/compairr_folder && \
    cp /home/ubuntu/compairr_folder/src/compairr /home/ubuntu/immuneML/compairr

# Install Python packages directly into system Python 3.11 (no venv needed in Docker)
RUN uv pip install --python python3.11 --system -r /home/ubuntu/immuneML/requirements-docker.txt
RUN uv pip install --python python3.11 --system --no-deps /home/ubuntu/immuneML/

# Download the IMGT human gene database for Stitchr
RUN stitchrdl -s human