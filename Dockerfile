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

# Installing yum dependencies 
RUN yum update -y
RUN yum install -y python38 python38-devel git dnf-plugins-core make openssl-devel libcurl-devel libxml2-devel gcc gcc-c++ autoconf automake libtool m4 llvm llvm-devel

# Installing the git dependency (no longer supported from pip>19)
RUN pip3 install -r requirements_DeepRC.txt

# Installing R
RUN dnf install -y epel-release
RUN dnf config-manager --set-enabled PowerTools
RUN yum install -y R

# Installing R dependencies
RUN R -e "install.packages('devtools', repos = 'https://cran.uib.no/')" -e "devtools::install_github('keshav-motwani/ggexp'); install.packages('Rmisc', dependencies = TRUE, repos = 'https://cran.uib.no/'); install.packages('readr', dependencies = TRUE, repos = 'https://cran.uib.no/')"

# Since we are not using venv's, we must install 'wheel' and 'setuptools' manually
RUN pip3 install wheel setuptools 

# Voila
RUN pip3 install ./immuneML/[all]
