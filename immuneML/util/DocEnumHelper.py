class DocEnumHelper:

    @staticmethod
    def get_enum_names(enum):

        options = str([option.name for option in enum])[1:-1].replace("'", "`")

        return options

    @staticmethod
    def get_enum_names_and_values(enum):

        options = str(["`" + option.name.lower() + "` (" + option.value + ")"
                       for option in enum])[1:-1].replace("'", "")

        return options
