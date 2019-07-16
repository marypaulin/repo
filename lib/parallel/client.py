from multiprocessing import Process
from os import kill
from signal import SIGINT
from time import sleep

class Client:
    def __init__(self, client_id, services, task):
        self.id = client_id
        self.pid = None
        self.process = Process(target=self.__run__, args=(client_id, services, task))
        self.process.daemon = True

    def __run__(self, client_id, services, task):
        try:
            for service in services:
                service.identify(client_id)
            task(client_id, services)
        except KeyboardInterrupt:
            pass
    
    def start(self):
        self.process.start()

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


    
