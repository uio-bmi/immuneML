FROM centos:latest

# Can't just copy '.', because it will copy '.git' folder too, which is ~600MB and we don't need it
COPY test ./immuneML/test/
COPY scripts ./immuneML/scripts/
COPY source ./immuneML/source/
COPY docs ./immuneML/docs/
COPY datasets ./immuneML/datasets/
COPY requirements.txt ./immuneML/
COPY setup.py ./immuneML/
COPY README.md ./immuneML/

# Enabling PowerTools
RUN yum update -y
RUN yum groupinstall -y 'Development Tools'
RUN dnf install -y epel-release
RUN yum install -y dnf-plugins-core
RUN dnf config-manager --set-enabled powertools

# Installing yum dependencies 
RUN yum install -y python38 python38-devel git libgit2 libgit2-devel make openssl-devel libcurl-devel libxml2-devel gcc gcc-c++ autoconf automake libtool m4 llvm llvm-devel cairo cairo-devel cairomm-devel libXt-devel

# Installing R
RUN yum install -y R

# Installing R dependencies
RUN R -e "install.packages('devtools', repos = 'https://cran.uib.no/')" -e "devtools::install_github('keshav-motwani/ggexp'); install.packages('Rmisc', dependencies = TRUE, repos = 'https://cran.uib.no/'); install.packages('readr', dependencies = TRUE, repos = 'https://cran.uib.no/')"

# Since we are not using venv's, we must install 'wheel' and 'setuptools' manually
RUN pip3 install wheel setuptools 

# Voila
RUN pip3 install ./immuneML/[all]
