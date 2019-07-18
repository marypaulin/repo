from multiprocessing import Process, Value
from threading import Thread, Event
from os import system, getpid
from subprocess import check_call, DEVNULL, STDOUT

def Client(id, services, task, client_type='process'):
    return Actor(id, services, task, actor_type=client_type)

def Server(id, services, server_type='process'):
    return Actor(id, services, __server_task__, actor_type=server_type)

def LocalClient(id, services, task):
    return Actor(id, services, task, actor_type='local')

def LocalServer(id, services):
    return Actor(id, services, __server_task__, actor_type='local')

def __server_task__(id, services, termination):
    while not termination():
        for service in services:
            service.serve()

def Actor(id, services, task, actor_type='process'):
    if actor_type == 'process':
        return __ProcessActor__(id, services, task)
    elif actor_type == 'thread':
        return __ThreadActor__(id, services, task)
    elif actor_type == 'local':
        return __LocalActor__(id, services, task)
    else:
        raise Exception("ActorException: Invalid Actor Type {}".format(actor_type))

class __LocalActor__:
    def __init__(self, id, services, task):
        self.id = id
        self.services = services
        self.task = task
        self.actor = lambda: task(id, services, lambda: False)

    def start(self, block=True):
        self.actor()

    def stop(self, block=True):
        return

    def join(self):
        return

    def is_alive(self):
        return False

class __ProcessActor__:
    def __init__(self, id, services, task):
        self.__termination__ = Value('d', 0)
        self.id = id
        self.actor = Process(target=self.__run__, args=(self.id, services, task, lambda: self.__termination__.value == 1))
        self.actor.daemon = True
        self.exception = None

    def __run__(self, id, services, task, termination):
        # Attempt to pin process to CPU core using taskset if available
        taskset_enabled = (system("command -v taskset") != 256)
        if taskset_enabled:
            check_call(["taskset", "-cp", str(id), str(getpid())], stdout=DEVNULL, stderr=STDOUT)
        try:
            task(id, services, termination)
        except Exception as e:
            self.exception = e
        finally:
            for service in services:
                service.close()

    def start(self, block=False):
        self.actor.start()
        while not self.is_alive():
            pass

    def stop(self, block=True):
        self.__termination__.value = 1
        if block:
            self.join()

    def join(self):
        self.actor.join()

    def is_alive(self):
        return self.actor.is_alive()


class __ThreadActor__:
    def __init__(self, id, services, task):
        self.__termination__ = Event()
        self.id = id
        self.actor = Thread(target=self.__run__, args=(self.id, services, task, self.__termination__.isSet))
        self.actor.daemon = True
        self.exception = None
        self.alive = Event()

    def __run__(self, id, services, task, termination):
        self.alive.set()
        try:
            task(id, services, termination)
        except Exception as e:
            self.exception = e
        finally:
            for service in services:
                service.close()
            self.alive.clear()

    def start(self, block=False):
        self.actor.start()
        while not self.is_alive():
            pass

    def stop(self, block=True):
        self.__termination__.set()
        if block:
            self.join()

    def join(self):
        self.actor.join()

    def is_alive(self):
        return self.alive.isSet()
