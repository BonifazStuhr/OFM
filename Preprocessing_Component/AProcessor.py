from abc import ABCMeta, abstractmethod

class AProcessor(metaclass=ABCMeta):
    """
    The AProcessor provides the interface for processing data

    :Attributes:
        input_name:    (String) The name of the input to apply this operation.
        output_names:  (String) The name of the output where this operation was applied.
    """
    def __init__(self, input_name, output_names):
        """
        Constructor, initialize member variables.
        """
        self.input_name = input_name
        self.output_names = output_names

    @abstractmethod
    def process(self, input):
        """
        Interface Method: The function (or graph part) of the processor.
        This function is the place to implement the processor logic.
        """
        raise NotImplementedError('Not implemented')



