import csv
import os


class OptimizationLogger:
    """
    A class for logging optimization iteration information and generating reports.

    Attributes
    ----------
    path : str
        File path to the directory where log files and reports are saved.
    verbose : boolean
        Flag to print logs to standard output.
    log_file : str
        File path of log file.
    history : list
        List to store iteration history.

    Parameters
    ----------
    path : str, optional
        File path to the directory where log files and reports are saved. Default is None.
    verbose : boolean, optional
        Flag to print logs to standard output. Default is False.
    """

    def __init__(self, path=None, verbose=False):
        self.path = path
        if self.path is None:
            self.path = ""
        else: 
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            if not self.path.endswith('/'):
                self.path += '/'
        self.verbose = verbose
        self.log_file = self.path + 'log_file.txt'
        self.history = []

    def log_iteration(self, iteration_info):
        """
        Log iteration information.

        Parameters
        ----------
        iteration_info : dict
            Dictionary containing iteration details.
        """
        self.history.append(iteration_info)

        with open(self.log_file, 'a+') as file:
            for k, v in iteration_info.items():
                if type(v) == float:
                    file.write(f"{v:<18.5E} ")
                elif type(v) == str:
                    file.write(f"{v:<18s} ")
                elif type(v) == bool:
                    file.write(f"{str(v):<18s} ")
                else:
                    file.write(f"{v:<18d} ")
            file.write('\n')

        if self.verbose:
            self.print_iteration(-1)

    def log_message(self, msg):
        """
        Log a message to the log file.

        Parameters
        ----------
        msg : str
            The message to be logged.
        """
        with open(self.log_file, 'a+') as f:
            f.write(f"{msg}\n")

        if self.verbose:
            print(msg)

    def print_iteration(self, i):
        """
        Print the details of a specific iteration.

        Parameters
        ----------
        i : int
            The index of the iteration to be printed.
        """
        info = self.history[i]
        for k, v in info.items():
            if type(v) == float:
                print(f"{v:<18.5E} ", end="")
            elif type(v) == str:
                print(f"{v:<18s} ", end="")
            elif type(v) == bool:
                print(f"{str(v):<18s} ", end="")
            else:
                print(f"{v:<18d} ", end="")
        print()
        # print('-' * 20)

    def print_history(self):
        """
        Print the full optimization history.
        """
        for info in self.history:
            for k, v in info.items():
                if type(v) == float:
                    print(f"{v:<18.5E} ", end="")
                elif type(v) == str:
                    print(f"{v:<18s} ", end="")
                elif type(v) == bool:
                    print(f"{str(v):<18s} ", end="")
                else:
                    print(f"{v:<18d} ", end="")
            print()

