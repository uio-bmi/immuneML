class MotifInstance:

    def __init__(self, instance: str, gap: int):
        assert gap == 0 or gap > 0 and instance.find("/") != -1, "MotifInstance: gap position is not indicated in the motif instance string. The " \
                                                                 "gap will be inserted in place of '/' sign. Check if there is a '/' sign in the " \
                                                                 "instance. "
        self.instance = instance
        self.gap = gap

    def __str__(self):
        return self.instance.replace("/", "".join(["/" for _ in range(self.gap)]))
