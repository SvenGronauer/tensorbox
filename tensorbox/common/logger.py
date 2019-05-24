import os
from abc import ABC, abstractmethod


class LoggerBase(ABC):
    def __init__(self, directory, stdout):
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.stdout = stdout
        # file_path = os.path.join(dir, 'summary.csv')
        self.file = None
        self.keys = ['global_step']
        self.sep = ','
        self.step = 0
    
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

    @abstractmethod
    def write(self, kvs, step):
        pass


class ConsoleLogger(LoggerBase):
    def __init__(self):
        directory = None
        super(ConsoleLogger, self).__init__(directory, stdout=True)

    def write(self, kvs, step):
        self.print_stdout(kvs, step)

    def close(self):
        pass


class CSVLogger(LoggerBase):
    def __init__(self, directory, stdout=True):
        super(CSVLogger, self).__init__(directory, stdout)
        file_path = os.path.join(directory, 'summary.csv')
        self.file = open(file_path, 'w+t')

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

        if self.stdout:
            s = step if step else self.step
            self.print_stdout(kvs, s)
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

