from immuneML.environment.Constants import Constants


__version__ = Constants.VERSION

try:
    import bionumpy.config
    bionumpy.config.LAZY = False
except ImportError:
    pass
