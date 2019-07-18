from multiprocessing import Process
from os import kill
from time import sleep
from signal import SIGINT, signal

# Wrapper class around multiprocessing.Process
# This mainly provides convenience methods for the following
#  - Graceful shutdown by sending interprocess signals
#  - Optional synchronization on setup and teardown
#  - Standardized method calling across all data-structures for client-server pattern



class Server:
    interrupt = False
    def __init__(self, server_id, services):
        self.id = server_id
        self.process = Process(target=self.__run__, args=(server_id, services))
        self.process.daemon = True
        self.interrupt = False

    def __interrupt__(self, signal, frame):
        self.interrupt = True
    
    def __run__(self, server_id, services):
        signal(SIGINT, self.__interrupt__)
        try:
            while not self.interrupt:
                modified = False
                for service in services:
                    modified = modified or service.serve()
                if not modified:
                    print("Server {} Idle".format(server_id))
                print("Foo")
        finally:
            for service in services:
                service.close(block=False)

    def start(self, block=True):
        self.process.start()
        while not self.is_alive():
            sleep(0.1)

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
