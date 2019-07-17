from multiprocessing import Lock, Pipe
from queue import Queue
from queue import Empty
from threading import Thread
# from os import close

# Wrapper class around the multiprocessing Pipe class made to resemble the multiprocessing Queue class
# This implementation was created because we 
class Channel:
    def __init__(self, multiproducer=True, multiconsumer=True):
        self.role = 'unidentified'
        self.buffer = Queue()
        self.inbound_lock = Lock() if multiproducer else None
        self.outbound_lock = Lock() if multiconsumer else None
        (self.outbound_connection, self.inbound_connection) = Pipe(True)
        self.flushing = False
        self.thread = None

    def identify(self, role):
        self.role = role
        # if self.role == 'producer':
        #     self.outbound_lock = None
        #     self.outbound_connection = None
        # elif self.role == 'consumer':
        #     self.inbound_lock = None
        #     self.inbound_connection = None
        # elif self.role == 'both':
        #     pass
        # else:
        #     raise Exception('ChannelError: Invalid role {}'.format(self.role))

    def pop(self, block=True):
        if not self.role in ('consumer', 'both'):
            raise Exception('ChannelError: Unauthorized Pop Operation from Role {}'.format(self.role))
        if self.outbound_connection == None:
            raise Exception('ChannelError: Outbound Connection Closed')
        poll_limit = None if block else 0
        element = None
        if self.outbound_lock != None:
            with self.outbound_lock:
                if self.outbound_connection.poll(poll_limit):
                    try:
                        element = self.outbound_connection.recv()
                    except (Exception, EOFError) as e:
                        pass
        else:
            if self.outbound_connection.poll(poll_limit):
                try:
                    element = self.outbound_connection.recv()
                except (Exception, EOFError) as e:
                    pass
        return element

    def __flush__(self):
        self.flushing = True
        while self.outbound_connection != None:
            try:
                element = self.buffer.get(block=False)
                self.__push__(element)
            except Empty:
                break
        self.flushing = False

    def __push__(self, element):
        if self.inbound_lock != None:
            with self.inbound_lock:
                try:
                    self.inbound_connection.send(element)
                except BrokenPipeError:
                    self.close_inbound()
        else:
            try:
                self.inbound_connection.send(element)
            except BrokenPipeError:
                    self.close_inbound()

    def push(self, element, block=True):
        if not self.role in ('producer', 'both'):
            raise Exception('ChannelError: Unauthorized Push Operation from Role {}'.format(self.role))
        if self.inbound_connection == None:
            raise Exception('ChannelError: Inbound Connection Closed')
        self.buffer.put(element)
        if not self.flushing:
            self.thread = Thread(target=self.__flush__)
            self.thread.daemon = True
            self.thread.start()
        if block and self.thread != None:
            self.thread.join()

    def __str__(self):
        return 'Channel({} => {})'.format(self.inbound_connection != None, self.outbound_connection != None)

    def __close_connection__(self, connection, lock=None):
        if lock != None:
            with lock:
                # close(connection.fileno())
                connection.close()
        else:
            # close(connection.fileno())
            connection.close()

    def close_inbound(self):
        if self.inbound_connection != None:
            if self.thread != None:
                self.thread.join()
            self.__close_connection__(self.inbound_connection, lock=self.inbound_lock)
            self.inbound_connection = None

    def close_outbound(self):
        if self.outbound_connection != None:
            while self.pop(block=False) != None:
                pass
            self.__close_connection__(self.outbound_connection, lock=self.outbound_lock)
            self.outbound_connection = None

    def close(self):
        if self.role == 'producer':
            self.close_inbound()
        elif self.role == 'consumer':
            self.close_outbound()
        else:
            self.close_inbound()
            self.close_outbound()
