FROM ubuntu:20.04

# Copy files
COPY . immuneML

RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install python3.8 python3-pip git-all -y

# install the dependency CompAIRR
RUN git clone https://github.com/uio-bmi/compairr.git compairr_folder
RUN make -C compairr_folder
RUN cp ./compairr_folder/src/compairr ./compairr

# Voila: install immuneML
RUN pip3 install ./immuneML/[TCRdist,gen_models,ligo]
