FROM ubuntu:22.04

# Copy files
COPY pyproject.toml /home/ubuntu/immuneML/
COPY requirements.txt /home/ubuntu/immuneML/
COPY requirements_TCRdist.txt /home/ubuntu/immuneML/
COPY requirements_DeepRC.txt /home/ubuntu/immuneML/
COPY README.md /home/ubuntu/immuneML/
COPY immuneML /home/ubuntu/immuneML/immuneML
COPY test /home/ubuntu/immuneML/test
COPY scripts /home/ubuntu/immuneML/scripts

RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git-all gcc g++ make python3.11 parasail python3.11-venv autotools-dev autoconf libtool pkgconf python3-dev

# install the dependency CompAIRR
RUN git clone https://github.com/uio-bmi/compairr.git /home/ubuntu/compairr_folder
RUN make -C /home/ubuntu/compairr_folder
RUN cp /home/ubuntu/compairr_folder/src/compairr /home/ubuntu/immuneML/compairr

# Voila: install immuneML in the virtual environment under /home/ubuntu/immuneML/.venv
RUN python3.11 -m venv /home/ubuntu/immuneML/.venv &&   \
    /home/ubuntu/immuneML/.venv/bin/python -m pip install --upgrade pip

RUN /home/ubuntu/immuneML/.venv/bin/python -m pip install /home/ubuntu/immuneML/[all]

# download the database to be able to export full-length sequences using Stitchr package
ENV PATH=/home/ubuntu/immuneML/.venv/bin:$PATH
RUN stitchrdl -s human

