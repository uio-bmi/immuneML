Running immuneML in the cloud
==============================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML in the cloud
   :twitter:description: See tutorials on how to run immuneML in the cloud.
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png



In this tutorial, we will show how immuneML can be run on Google Cloud and Amazon Web Services (AWS) using the existing Docker image.

Installing immuneML on Google Cloud
-------------------------------------

Here we will describe how immuneML can be set up on a virtual machine on Google Cloud, so that it is possible to log in to the machine and perform an
analysis using immuneML.

To follow this tutorial, you will need an account on `Google Cloud <https://cloud.google.com>`_.

The following video shows how to set up immuneML once you have a Google Cloud account.

.. raw:: html

  <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/hpxvfvN83g8" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

immuneML Docker image is available at `Docker Hub <https://hub.docker.com/repository/docker/milenapavlovic/immuneml>`_.

Installing immuneML on AWS
---------------------------

Here we will describe how immuneML can be set up on AWS. We will use a single EC2 Linux instance and install
immuneML using Docker. Once the instance is created, the process is the same as for any other installation of a Docker image on a Linux machine.

To install immuneML on AWS, follow these steps:

1. Set up to use Amazon EC2 instance (a single virtual machine) following the steps described `here <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/get-set-up-for-amazon-ec2.html>`_ including signing up for AWS and creating a key
pair and a security group for accessing virtual machines on AWS.

2. Launch an EC2 instance following `this tutorial <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html>`_. When choosing the
instance type, for running immuneML you will need at least 4GB of RAM and 30GB of storage (but more is recommended, depending on the analyses you plan to run).

3. Follow `this tutorial <https://docs.docker.com/engine/install/ubuntu/>`_ to install Docker on Ubuntu (or choose an alternative operating system
from the side menu if you choose a different operating system for your EC2 instance).

4. Once the Docker is installed, install immuneML Docker image and run a quickstart analysis:

.. code-block:: console

  docker run -it -v $(pwd):/data --name my_container milenapavlovic/immuneml immune-ml-quickstart /data/output/

This command will download the immuneML Docker image, start a Docker container called :code:`my_container` from the downloaded image, run the quickstart, and store results in :code:`output` directory in the current directory.
For more information on immuneML and Docker, see the :ref:`Setting up immuneML with Docker`.