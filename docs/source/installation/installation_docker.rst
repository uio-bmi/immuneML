Setting up immuneML with Docker
================================

This tutorial assumes you have Docker installed on your machine. To install it, see `the official Docker documentation <https://docs.docker.com/get-docker/>`_.

To test that your docker works you can use the command:

.. code-block:: console

  docker run hello-world

This should give the hello world message if it is working properly.

.. code-block:: console

  Hello from Docker!
  This message shows that your installation appears to be working correctly.

To get the immuneml docker container use the command (you might have to use sudo).

.. code-block:: console

  docker pull milenapavlovic/immuneml

To confirm that the image is available run:

.. code-block:: console

  docker images

You should see milenapavolvic/immuneml in the list of images.

You can confirm that immuneml is working properly by running the tests:

.. code-block:: console

  docker run milenapavlovic/immuneml pytest immuneML/test

When you run the docker container you may want to use your local datafiles for an analysis with immuneml. To achieve this you can use the command

.. code-block:: console

  docker run -ti -v $(pwd):/app  milenapavlovic/immuneml

This will bind your current working directory to the path /app inside the docker container and start the container. Your local files will be visible
in the container, and any data that you put under /app will exist after you close the docker instance with the `exit` command.

