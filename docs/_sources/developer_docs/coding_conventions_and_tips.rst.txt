.. note::
  **Coding conventions and tips**

  #. Class names are written in CamelCase
  #. Class methods are writte in snake_case
  #. Abstract base classes :code:`MLMethod`, :code:`DatasetEncoder`, and :code:`Report`, define an interface for their inheriting subclasses. These classes contain abstract methods which should be implemented.
  #. Class methods starting with _underscore are generally considered "private" methods, only to be called by the class itself. If a method is expected to be called from another class, the method name should not start with an underscore.
  #. When familiarising yourself with existing code, we recommend focusing on public methods. Private methods are typically very unique to a class (internal class-specific calculations), whereas the public methods contain more general functionalities (e.g., returning a main result).
  #. If your class should have any default parameters, they should be defined in a default parameters file under :code:`config/default_params/`.
  #. Some utility classes are available in the :code:`util` package to provide useful functionalities. For example, :py:obj:`~immuneML.util.ParameterValidator.ParameterValidator` can be used to check user input and generate error messages, or :py:obj:`~immuneML.util.PathBuilder.PathBuilder` can be used to add and remove folders.