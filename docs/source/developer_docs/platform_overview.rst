immuneML platform overview
============================

An overview of the most used packages with their dependencies in immuneML is shown in the figure below. However, to extend the platform, it is only
necessary to follow the tutorials without the need to go into all platform details.

.. figure:: ../_static/images/dev_docs/immuneML_packages.png
  :width: 70%
  :alt: UML diagram of the immuneML's packages and dependencies between them

  UML diagram showing the immuneML packages and the dependencies between them

For more details on the data model, see :ref:`immuneML data model`.

Extending the platform
---------------------------

The tutorials provided in the documentation focus on adding new machine learning methods, encodings and analysis reports. The relevant architecture is
shown in the diagram below.

.. figure:: ../_static/images/dev_docs/extending_immuneML.png
  :width: 70%
  :alt: UML diagram showing existing components and how the platform could be extended

  UML diagram showing existing components and how the platform could be extended by implementing functionalities defined by corresponding abstract
  classes