from multiprocessing import Process, Value
from threading import Thread, Event
from os import system, getpid
from subprocess import check_call, DEVNULL, STDOUT
from traceback import print_exc


def Client(id, services, task, client_type='process', peers=None):
    return Actor(id, services, task, actor_type=client_type, peers=peers)

def Server(id, services, server_type='process', peers=None):
    return Actor(id, services, __server_task__, actor_type=server_type, peers=peers)

def LocalClient(id, services, task, peers=None):
    return Actor(id, services, task, actor_type='local', peers=peers)

def LocalServer(id, services, peers=None):
    return Actor(id, services, __server_task__, actor_type='local', peers=peers)

def __server_task__(id, services, peers):
    while peers.value > 0: # Continue servicing as it is not alone
        for service in services:
            service.serve()

def __tear_down__(services, peers):
    if peers != None:
        with peers.get_lock():
            peers.value -= 1
        while peers.value > 0:
            for service in services:
                service.flush()

def Actor(id, services, task, actor_type='process', peers=None):
    if actor_type == 'process':
        return __ProcessActor__(id, services, task, peers=peers)
    elif actor_type == 'thread':
        return __ThreadActor__(id, services, task, peers=peers)
    elif actor_type == 'local':
        return __LocalActor__(id, services, task, peers=peers)
    else:
        raise Exception("ActorException: Invalid Actor Type {}".format(actor_type))

class __LocalActor__:
    def __init__(self, id, services, task, peers=None):
        self.id = id
        self.services = services
        self.task = task
        self.__run__ = lambda: task(id, services, peers)
        self.services = services
        self.peers = peers

    def start(self, block=False):
        self.__run__()
        __tear_down__(self.services, self.peers)
    def join(self):
        return

    def is_alive(self):
        return False

class __ProcessActor__:
    def __init__(self, id, services, task, peers=None):
        self.id = id
        self.actor = Process(target=self.__run__, args=(self.id, services, task, peers))
        self.actor.daemon = True
        self.exception = None

    def __run__(self, id, services, task, peers):
        # Attempt to pin process to CPU core using taskset if available
        taskset_enabled = (system("command -v taskset") != 256)
        if taskset_enabled:
            check_call(["taskset", "-cp", str(id), str(getpid())], stdout=DEVNULL, stderr=STDOUT)
        try:
            task(id, services, peers)
        except Exception as e:
            self.exception = e
            print_exc()
            print("ActorException: Actor ID {} caught: {}".format(id, e))
        finally:
            __tear_down__(services, peers)


    def start(self, block=False):
        self.actor.start()
        while block and not self.is_alive():
            pass

    def join(self):
        self.actor.join()

    def is_alive(self):
        return self.actor.is_alive()

class __ThreadActor__:
    def __init__(self, id, services, task, peers=None):
        self.id = id
        self.actor = Thread(target=self.__run__, args=(self.id, services, task, peers))
        self.actor.daemon = True
        self.exception = None
        self.alive = Event()

    def __run__(self, id, services, task, peers):
        self.alive.set()
        try:
            task(id, services, peers)
        except Exception as e:
            self.exception = e
            print_exc()
            print("ActorException: Actor ID {} caught: {}".format(id, e))
        finally:
            __tear_down__(services, peers)
            self.alive.clear()

    def start(self, block=False):
        self.actor.start()
        while block and not self.is_alive():
            pass

    def join(self):
        self.actor.join()

    def is_alive(self):
        return self.alive.isSet()
