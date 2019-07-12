import os
import sys
import time
from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import logging

np.set_printoptions(precision=4)


class LoggerBase(ABC):
    def __init__(self, log_dir, file_name, stdout=True, *args, **kwargs):
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

    def print_stdout(self, key_values, step):
        if not self.stdout:
            return
        string = 'Epoch: {}'.format(step)
        for k, v in key_values.items():
            # if isinstance(v, float):
            string += '\t {}: {:0.3f}'.format(k, float(v))
            # else:
            #     print(type(v))
            #     string += '\t {}: {}'.format(k, v)
        print(string)

    def warning(self, msg):
        self.logger.warning(msg)

    @abstractmethod
    def write(self, key_values, step):
        pass


class ConsoleLogger(LoggerBase):
    def __init__(self):

        log_dir = None
        super(ConsoleLogger, self).__init__(log_dir, file_name='ConsoleLogger', stdout=True)

    def write(self, key_values, step):
        self.print_stdout(key_values, step)

    def close(self):
        pass


class CSVLogger(LoggerBase):
    """ source: OpenAI Baselines GitHub
        https://github.com/openai/baselines
    """
    def __init__(self,
                 log_dir,
                 file_name='summary.csv',
                 *args,
                 **kwargs):
        super(CSVLogger, self).__init__(log_dir, file_name, *args, **kwargs)
        file_path = os.path.join(self.log_dir, file_name)
        self.file = open(file_path, 'w+t')

    def write(self, key_values, step=None):
        # Add our current row to the history
        extra_keys = list(key_values.keys() - self.keys)
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
            self.print_stdout(key_values, s)
        if step:
            self.file.write(str(step))
        else:
            self.step += 1
            self.file.write(str(self.step))
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            v = key_values.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write('\n')
        self.file.flush()


class TensorBoardLogger(LoggerBase):
    def __init__(self, log_dir, *args, **kwargs):
        super(TensorBoardLogger, self).__init__(log_dir, file_name='TensorBoardLogger', *args, **kwargs)
        self.writer = tf.summary.create_file_writer(self.log_dir)
        print('[TensorBoardLogger] Create TF event files at:', self.log_dir)

    def write(self, key_values, step):
        self.print_stdout(key_values, step)
        with self.writer.as_default():
            for key, value in key_values.items():
                tf.summary.scalar(key, value, step=step)

    def close(self):
        pass


class CombinedLogger(LoggerBase):
    available_loggers = {
        'CSV': CSVLogger,
        'TensorBoard': TensorBoardLogger
    }

    def __init__(self, log_dir, loggers=('CSV', 'TensorBoard')):
        self.loggers = []
        for logger in loggers:
            if logger in CombinedLogger.available_loggers:
                instance = CombinedLogger.available_loggers[logger](log_dir=log_dir, stdout=False)
                self.loggers.append(instance)
        print('Number of loggers:', len(self.loggers), 'Names:', loggers)
        super(CombinedLogger, self).__init__(log_dir, file_name='CombinedLogger', stdout=True)

    def write(self, key_values, step):
        self.print_stdout(key_values, step)
        for logger in self.loggers:
            logger.write(key_values, step)

    def close(self):
        pass



if __name__ == '__main__':
    size = 500

