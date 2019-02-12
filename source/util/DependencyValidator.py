class DependencyValidator:
    """
    Even when there are no circular dependencies, if imports are
    circular, Python cannot handle that (e.g. when an instance of
    the signal class passes itself to SignalImplantingStrategy -
    some sort of inversion of control). This class then performs
    that custom type-checking where needed.
    """
    @staticmethod
    def check_signal(instance):
        return instance.__class__.__name__ == "Signal"
