import os
import sys
import time
from abc import ABC, abstractmethod

import numpy as np
import logging


class LoggerBase(ABC):
    def __init__(self, log_dir, file_name, stdout=True):
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        self.file_name = file_name
        os.makedirs(log_dir, exist_ok=True)
        self.stdout = stdout
        self.file = None
        self.keys = ['global_step']
        self.sep = ','
        self.step = 0
        self.warning('There has been no logger defined')
    
    def close(self):
        if self.file:
            self.file.close()
        else:
            raise ValueError('No file specified.')

    def dump_as_json(self, dic):
        pass

    @staticmethod
    def print_stdout(kvs, step):
        string = 'Epoch: {}'.format(step)
        for k, v in kvs.items():
            if isinstance(v, float):
                string += '\t {}: {:0.3f}'.format(k, v)
            else:
                string += '\t {}: {}'.format(k, v)
        print(string)

    def warning(self, msg):
        self.logger.warning(msg)

    @abstractmethod
    def write(self, kvs, step):
        pass


class ConsoleLogger(LoggerBase):
    def __init__(self):
        log_dir = None
        super(ConsoleLogger, self).__init__(log_dir, file_name='ConsoleLogger', stdout=True)

    def write(self, kvs, step):
        self.print_stdout(kvs, step)

    def close(self):
        pass


class CSVLogger(LoggerBase):
    """ source: OpenAI Baselines GitHub
        https://github.com/openai/baselines
    """
    def __init__(self,
                 log_dir,
                 total_steps,
                 file_name='summary.csv',
                 **kwargs):
        super(CSVLogger, self).__init__(log_dir, file_name, **kwargs)
        file_path = os.path.join(self.log_dir, file_name)
        self.file = open(file_path, 'w+t')
        self.progress_bar = plot_progress
        self.total_steps = total_steps

    def write(self, kvs, step=None):
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:  # add missing keys to dictionary and file
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)

            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(k)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')

        s = step if step else self.step
        if self.stdout:  # print each step in shell
            self.print_stdout(kvs, s)
        else:  # show progress bar in shell
            plot_progress(s, block_size=1, total_size=self.total_steps)
            if step >= (self.total_steps-1):
                # sys.stdout.flush()
                sys.stdout.write('\x1b[2K')  # erases the plot progress line
                sys.stdout.write('\n')
                self.print_stdout(kvs, s)  # show finals values
        if step:
            self.file.write(str(step))
        else:
            self.step += 1
            self.file.write(str(self.step))
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write('\n')
        self.file.flush()


class Progbar(object):
    """ Displays a progress bar.
        Source: tf.keras.utils.Progbar
    """
    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None, unit_name='step'):
        """

        :param target: Total number of steps expected, None if unknown.
        :param width: Progress bar width on screen.
        :param verbose:  Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        :param interval: Minimum visual progress update interval (in seconds).
        :param stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over time. Metrics in this list
          will be displayed as-is. All others will be averaged
          by the progbar before display.
        :param unit_name: Display name for step counts (usually "step" or "sample").
        """
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
        current: Index of current step.
        values: List of tuples:
            `(name, value_for_last_step)`.
            If `name` is in `stateful_metrics`,
            `value_for_last_step` will be displayed as-is.
            Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is not None and current >= self.target:
                numdigits = int(np.log10(self.target)) + 1
                count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


class ProgressTracker(object):
    """ source: tf.keras.utils.Progbar """
    # Maintain bar for the lifetime of download.
    bar = None


def plot_progress(count, block_size, total_size):
    if ProgressTracker.bar is None:
        if total_size == -1:
            total_size = None
        ProgressTracker.bar = Progbar(total_size)
    else:
        ProgressTracker.bar.update(count * block_size)
    if count + block_size >= total_size:
        ProgressTracker.bar = None  # reset the status


if __name__ == '__main__':
    size = 500
    for i in range(500):
        plot_progress(i, 1, size)
        time.sleep(0.03)

