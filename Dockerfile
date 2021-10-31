FROM centos:latest

# Copy files
COPY . immuneML

# Configurinig yum
RUN yum update -y
RUN yum groupinstall -y 'Development Tools'
RUN yum install -y dnf-plugins-core

# Enabling PowerTools
RUN dnf install -y epel-release
RUN dnf config-manager --set-enabled powertools

# Installing yum dependencies 
RUN yum install -y python38 python38-devel git libgit2 libgit2-devel make openssl-devel libcurl-devel libxml2-devel gcc gcc-c++ autoconf automake libtool m4 llvm llvm-devel cairo cairo-devel cairomm-devel libXt-devel

# Since we are not using venv's, we must install 'wheel' and 'setuptools' manually
RUN pip3 install wheel setuptools

# install the dependency CompAIRR
RUN yum install git -y
RUN git clone https://github.com/uio-bmi/compairr.git compairr_folder
RUN make -C compairr_folder
RUN cp ./compairr_folder/src/compairr ./compairr

# Voila: install immuneML
RUN pip3 install ./immuneML/[TCRdist]
