from multiprocessing import Process
from os import kill
from signal import SIGINT
from time import sleep

# Wrapper class around multiprocessing.Process
# This mainly provides convenience methods for the following
#  - Graceful shutdown by sending interprocess signals
#  - Optional synchronization on setup and teardown
#  - Standardized method calling across all data-structures for client-server pattern
class Server:
    def __init__(self, server_id, services):
        self.id = server_id
        self.pid = None
        self.process = Process(target=self.__run__, args=(server_id, services))
        self.process.daemon = True

    def __run__(self, server_id, services):
        terminate = False
        while not terminate:
            try:
                for service in services:
                    service.serve()
            except KeyboardInterrupt:
                terminate = True
        

    
    def start(self, block=True):
        self.process.start()
        self.pid = self.process.pid
        while not self.is_alive():
            sleep(0.1)

    def stop(self, block=True):
        kill(self.pid, SIGINT)
        if block:
            self.join()

    def join(self):
        self.process.join()

    def is_alive(self):
        return self.process.is_alive()
