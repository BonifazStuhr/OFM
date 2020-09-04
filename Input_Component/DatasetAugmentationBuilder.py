class DatasetAugmentationBuilder:
    """
    The DatasetAugmentationBuilder builds a pipeline for augmentation from the given preprocessors, generators and
    output_preprocessors

    :Attributes:
        preprocessors:          (List of (tf) functions) preprocessors will be applied first and create no "new data"
                                they only change the input data by preprocessing.
        generators:             (List of (tf) functions) generators will be applied second and create "new data"
                                by changing the given date from input or preprocessing.
        output_preprocessors:   (List of (tf) functions) output_preprocessors will be applied last and create no "new data"
                                they only change the data from the generators or preprocessors or the input.

    """
    def __init__(self, preprocessors=None, generators=None, output_preprocessors=None):
        """
        Constructor, initialize member variables.
        :param preprocessors: (List of (tf) functions) preprocessors will be applied first and create no "new data"
                              they only change the input data by preprocessing. None by default.
        :param generators:  (List of (tf) functions) generators will be applied second and create "new data"
                             by changing the given date from input or preprocessing. None by default.
        :param output_preprocessors: (List of (tf) functions) output_preprocessors will be applied last and create no "new data"
                                     they only change the data from the generators or preprocessors or the input. None by default.

         """
        self.preprocessors = preprocessors
        self.generators = generators
        self.output_preprocessors = output_preprocessors

    def generate_sample_output(self, data):
        """
        Generates the output by applying the preprocessors, generators, output_preprocessors.
        :param data: (Tensor) The data to augment.
        :return: data: (Tensor) The augmented data.
        """
        data = self.runProcessors(self.preprocessors, data)
        data = self.runProcessors(self.generators, data)
        data = self.runProcessors(self.output_preprocessors, data)
        return data

    def runProcessors(self, processors, data):
        """
        Generates the output by applying the given processors.
        :param processors: (List of (tf) functions) the processors to augment the data with.
        :param data: (Tensor) The data to augment.
        :return: data: (Tensor) The augmented data.
        """
        if processors:
            for processor in processors:
                processor_outputs, output_names = processor.process(data)
                for output, name in zip(processor_outputs, output_names):
                    data[name] = output
        return data