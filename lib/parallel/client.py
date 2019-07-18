from multiprocessing import Process
from os import kill, system, getpid
from signal import SIGINT
from time import sleep

class Client:
    def __init__(self, client_id, services, task):
        self.id = client_id
        self.process = Process(target=self.__run__, args=(client_id, services, task))
        self.process.daemon = True

    def __run__(self, client_id, services, task):

       taskset_enabled = (system("command -v taskset") != 256)
       # handler = signal.signal(signal.SIGINT, lambda signal_number, frame: self.terminate())

       commands = []
       for i in range(self.size):
            (process, _connection) = pool[i]
            process.start()
            if taskset_enabled:
                check_call(
                    [
                        "taskset",
                        "-cp",
                        str(i % os.cpu_count()),
                        str(process.pid)
                    ],
                    stdout=DEVNULL, stderr=STDOUT
                )

        try:
            task(client_id, services)
        finally:
            for service in services:
                service.close()
    
    def start(self):
        self.process.start()

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


    
