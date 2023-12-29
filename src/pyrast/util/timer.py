import time
import numpy as np


class Timer:
    class Context:
        def __init__(self, timer, name):
            self.timer = timer
            self.name = name

        def __enter__(self):
            self.timer.start(self.name)

        def __exit__(self, *args):
            self.timer.stop(self.name)

    def __init__(self):
        self.times = {}
        self.history = {}

    def __call__(self, name: str):
        """
        Creates timer context to simplify start and stop calls
        Args:
            name: Name of the section 

        Returns: context manager for section

        """
        return Timer.Context(self, name)

    def start(self, name: str):
        """
        Start timer
        Args:
            name: section name 
        """
        self.times[name] = time.perf_counter_ns()

    def stop(self, name: str):
        """
        Stop timer
        Args:
            name: section name 
        """
        assert name in self.times
        diff = time.perf_counter_ns() - self.times[name]

        if name not in self.history:
            self.history[name] = []
        self.history[name].append(diff)
        return diff

    def format_statistics(self, names=None):
        """
        Outputs timing statistics
        Args:
            names: list of section names or None 

        Returns:
            String containing summary statistics about the measurements

        """
        header = "{:>20} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}".format(
            "Section Name", "Total", "Mean", "Std", "Min", "Max"
        )
        fmt_string = "{:>20} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f}"
        if names is None:
            names = self.history.keys()
        output = [header]

        for name in names:
            # compute values in millisecond
            values = np.array(self.history[name]) / 1000000
            output.append(
                fmt_string.format(name, values.sum(),
                                  values.mean(), values.std(),
                                  values.min(), values.max())
            )
        return "\n".join(output)
