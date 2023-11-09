import time
import numpy as np


class Timer:
    class TimerContext:
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

    def section(self, name):
        return Timer.TimerContext(self, name)

    def start(self, name):
        self.times[name] = time.perf_counter_ns()

    def stop(self, name):
        assert name in self.times
        diff = time.perf_counter_ns() - self.times[name]

        if name not in self.history:
            self.history[name] = []
        self.history[name].append(diff)
        return diff

    def format_statistics(self, names=None):
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
