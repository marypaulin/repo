from multiprocessing import Lock, Pipe
from queue import Queue
from queue import Empty
from threading import Thread
# from os import close

# Wrapper class pair around the multiprocessing Pipe class made to resemble the multiprocessing Queue class
# This implementation exposes separate producer and consumer ends to prevent replication of excess file descriptors

def Channel(producers=1, consumers=1, duplex=False):
    if duplex:
        (connection_a, connection_b) = Pipe(True)

        consumer_a = __ChannelConsumer__(connection_a, lock=Lock() if producers > 1 else None)
        producer_a = __ChannelProducer__(connection_a, lock=Lock() if consumers > 1 else None)

        consumer_b = __ChannelConsumer__(connection_b, lock=Lock() if producers > 1 else None)
        producer_b = __ChannelProducer__(connection_b, lock=Lock() if consumers > 1 else None)

        endpoint_a = __ChannelProducerConsumer__(consumer_a, producer_a)
        endpoint_b = __ChannelProducerConsumer__(consumer_b, producer_b)
        return (endpoint_a, endpoint_b)
    else:
        (read_connection, write_connection) = Pipe(False)
        consumer = __ChannelConsumer__(write_connection, lock=Lock() if producers > 1 else None)
        producer = __ChannelProducer__(read_connection, lock=Lock() if consumers > 1 else None)
        return (consumer, producer)

def EndPoint(consumer, producer):
    return __ChannelProducerConsumer__(consumer, producer)

class __ChannelProducerConsumer__:
    def __init__(self, consumer, producer):
        self.consumer = consumer
        self.producer = producer

    def push(self, element, block=True):
        self.consumer.push(element, block=block)
    
    def pop(self, block=True):
        return self.producer.pop(block=block)

    def close(self, block=True):
        self.producer.close(block=block)
        self.consumer.close(block=block)

class __ChannelConsumer__:
    def __init__(self, connection, lock=None):
        self.lock = lock
        self.connection = connection
        self.buffer = Queue()
        self.flushing = False
        self.thread = None
    
    def push(self, element, block=True):
        if self.connection == None:
            raise Exception('ChannelError: Inbound Connection Closed')
        self.buffer.put(element)
        if not self.flushing:
            self.thread = Thread(target=self.__flush__)
            self.thread.daemon = True
            self.thread.start()
        if block and self.thread != None:
            self.thread.join()
    
    def close(self, block=True):
        if self.connection != None:
            self.connection.close()
            if block and self.thread != None:
                self.thread.join()
            self.connection = None

    def __flush__(self):
        self.flushing = True
        while self.connection != None:
            try:
                element = self.buffer.get(block=False)
                self.__push__(element)
            except Empty:
                break
            except OSError as e:
                self.buffer.put(element)
                break
        self.flushing = False
    
    def __push__(self, element):
        if self.lock != None:
            with self.lock:
                self.connection.send(element)
        else:
            self.connection.send(element)


class __ChannelProducer__:
    def __init__(self, connection, lock):
        self.lock = lock
        self.connection = connection
    

    def pop(self, block=True):
        if self.connection == None:
            raise Exception('ChannelError: Outbound Connection Closed')
        
        # Note: 'None' indicates indefinite waiting, '0' indicates no waiting
        timeout = None if block else 0

        element = None
        if self.lock != None:
            with self.lock:
                if self.connection.poll(timeout):
                    try:
                        element = self.connection.recv()
                    except EOFError:
                        pass
                    except Exception as e:
                        print(e)
                
        else:
            if self.connection.poll(timeout):
                try:
                    element = self.connection.recv()
                except EOFError:
                    pass
                except Exception as e:
                    print(e)                    
        return element

    def close(self, block=True):
        if self.connection != None:
            while block and self.pop(block=False) != None:
                pass
            if self.lock != None:
                with self.lock:
                    self.connection.close()
            else:
                self.connection.close()
            self.connection = None
