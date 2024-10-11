from bionumpy.encodings.string_encodings import StringEncoding


class NullableBoolStringEncoding(StringEncoding):

    returns_raw = True

    def __hash__(self):
        return hash("NullableBoolStringEncoding")


nullable_bool_string = NullableBoolStringEncoding(['', 'False', 'True'])
