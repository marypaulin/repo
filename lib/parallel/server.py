from multiprocessing import Process
from os import kill, system, getpid
from time import sleep
from signal import SIGINT, signal
from subprocess import check_call, DEVNULL, STDOUT
from threading import Thread

# Wrapper class around multiprocessing.Process
# This mainly provides convenience methods for the following
#  - Graceful shutdown by sending interprocess signals
#  - Optional synchronization on setup and teardown
#  - Standardized method calling across all data-structures for client-server pattern

class Server:
    def __init__(self, server_id, services):
        self.id = server_id
        self.process = Process(target=self.__run__, args=(server_id, services))
        self.process.daemon = True
        self.interrupt = False

    def __interrupt__(self, signal, frame):
        self.interrupt = True
    
    def __run__(self, server_id, services):
        # Attempt to pin process to CPU core using taskset if available
        taskset_enabled = (system("command -v taskset") != 256)
        if taskset_enabled:
            check_call(["taskset", "-cp", str(server_id), str(getpid())], stdout=DEVNULL, stderr=STDOUT)
        signal(SIGINT, self.__interrupt__)
        try:
            while not self.interrupt:
                modified = False
                for service in services:
                    modified = modified or service.serve()
                sleep(0)
        finally:
            for service in services:
                service.close(block=False)

    def start(self, block=True):
        self.process.start()
        while not self.is_alive():
            pass

    def stop(self, block=True):
        kill(self.process.pid, SIGINT)
        if block:
            self.join()
    
    def terminate(self):
        self.process.terminate()

    def join(self):
        self.process.join()

    def is_alive(self):
        return self.process.is_alive()


class ServerThread:
    def __init__(self, server_id, services):
        self.id = server_id
        self.thread = Thread(target=self.__run__, args=(server_id, services))
        self.thread.daemon = True
        self.interrupt = False

    def __run__(self, server_id, services):
        try:
            while not self.interrupt:
                modified = False
                for service in services:
                    modified = modified or service.serve()
                sleep(0)
        finally:
            for service in services:
                service.close(block=False)

    def start(self):
        self.thread.start()

    def stop(self, block=True):
        self.interrupt = True
        if block:
            self.thread.join()

    def terminate(self):
        self.interrupt = True

    def join(self):
        self.thread.join()
