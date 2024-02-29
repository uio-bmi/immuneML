Setting up immuneML with Docker
================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML: setting up with Docker
   :twitter:description: See tutorials on how to set up immuneML with Docker.
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png


This tutorial assumes you have Docker installed on your machine. To install it, see `the official Docker documentation <https://docs.docker.com/get-docker/>`_.

Getting started with immuneML and Docker
-----------------------------------------

Once you have Docker working on your machine, use the following command to download and run the Docker image with immuneML analysis. This will do the following:

  1. create the Docker container with the given name (here: :code:`my_container`),

  2. bind the current working directory to the path /data inside the container which will make the data from the working directory visible inside the container and which will keep the data placed there visible after the container is stopped,

  3. run an immuneML quickstart analysis and store the output in the new 'output' directory in the current working directory:

.. code-block:: console

  docker run -it -v $(pwd):/data --name my_container milenapavlovic/immuneml immune-ml-quickstart /data/output/

To run the analysis using a custom the specs.yaml file from the current working directory as analysis specification and store the results in the
'output_custom' directory which will be created in the current working directory, run the following command:

.. code-block:: console

  docker run -it -v $(pwd):/data --name my_container milenapavlovic/immuneml immune-ml /data/specs.yaml /data/output_custom/

To exit the Docker container, use the following command:

.. code-block:: console

  exit

Using the Docker container for longer immuneML runs
----------------------------------------------------
Ï
If you expect the analysis to take more time, you can start the container as a background process. The command to run in that case would be the following:

.. code-block:: console

  docker run -itd -v $(pwd):/data --name my_container milenapavlovic/immuneml immune-ml-quickstart /data/output/

To see the logs, run the following command with the container name (here: :code:`my_container`):

.. code-block:: console

  docker logs my_container

To see the list of available containers, you can use the following command:

.. code-block:: console

  docker ps -a

If you just started the container with the previous command, the output showing the list of available containers should look similar to this:

.. code-block:: console

  CONTAINER ID        IMAGE                     COMMAND             CREATED             STATUS              PORTS               NAMES
  e799e644e479        milenapavlovic/immuneml   "/bin/bash"         34 seconds ago      Up 33 seconds                           my_container

To stop the container, run the following command where the argument is the name of your container:

.. code-block:: console

  docker stop my_container

To delete the container, run the following command where the argument is the name of your container:

.. code-block:: console

  docker rm my_container


