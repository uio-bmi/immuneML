FROM ubuntu:20.04

# Copy files
COPY . immuneML

RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install python3.8 python3-pip git-all -y
RUN pip3 install setuptools==50.3.2 cython

# install the dependency CompAIRR
RUN git clone https://github.com/uio-bmi/compairr.git compairr_folder
RUN make -C compairr_folder
RUN cp ./compairr_folder/src/compairr ./compairr

# Voila: install immuneML
RUN pip3 install ./immuneML/[TCRdist]

# temporary fix for ValueError due to binary incompatibilities with numpy in cfisher package (dependency for computing fisher's exact test)
RUN pip3 install --upgrade numpy
