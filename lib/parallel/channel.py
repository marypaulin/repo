from multiprocessing import Lock, Pipe, Queue
from collections import deque
from threading import Thread
from queue import Full, Empty

# Wrapper class pair around the multiprocessing Pipe class made to resemble the multiprocessing Queue class
# This implementation exposes separate producer and consumer ends to prevent replication of excess file descriptors

def Channel(read_lock=False, write_lock=False, duplex=False, buffer_limit=None, channel_type=None):
    if channel_type == None:
        channel_type = 'queue' if read_lock or write_lock else 'pipe'
    if channel_type == 'pipe':
        return PipeChannel(read_lock=read_lock, write_lock=write_lock, duplex=duplex, buffer_limit=buffer_limit)
    elif channel_type == 'queue':
        return QueueChannel(read_lock=read_lock, write_lock=write_lock, duplex=duplex, buffer_limit=buffer_limit)
    else:
        raise Exception("ChannelException: Invalid Channel Type {}".format(channel_type))


def PipeChannel(read_lock=False, write_lock=False, duplex=False, buffer_limit=None):
    if duplex:
        (connection_a, connection_b) = Pipe(True)

        consumer_a = __ConnectionConsumer__(connection_a, lock=Lock() if read_lock or write_lock else None, buffer_limit=buffer_limit)
        producer_a = __ConnectionProducer__(connection_a, lock=Lock() if read_lock or write_lock else None)

        consumer_b = __ConnectionConsumer__(connection_b, lock=Lock() if read_lock or write_lock else None, buffer_limit=buffer_limit)
        producer_b = __ConnectionProducer__(connection_b, lock=Lock() if read_lock or write_lock else None)

        endpoint_a = __ProducerConsumer__(consumer_a, producer_a)
        endpoint_b = __ProducerConsumer__(consumer_b, producer_b)
        return (endpoint_a, endpoint_b)
    else:
        (read_connection, write_connection) = Pipe(False)
        consumer = __ConnectionConsumer__(write_connection, lock=Lock() if write_lock else None, buffer_limit=buffer_limit)
        producer = __ConnectionProducer__(read_connection, lock=Lock() if read_lock else None)
        return (consumer, producer)

def QueueChannel(read_lock=False, write_lock=False, duplex=False, buffer_limit=None):
    if duplex:
        if buffer_limit == None:
            forward_queue = Queue()
            backward_queue = Queue()
        else:
            forward_queue = Queue(buffer_limit)
            backward_queue = Queue(buffer_limit)

        consumer_a = __QueueConsumer__(forward_queue)
        producer_a = __QueueProducer__(backward_queue)

        consumer_b = __QueueConsumer__(backward_queue)
        producer_b = __QueueProducer__(forward_queue)

        endpoint_a = __ProducerConsumer__(consumer_a, producer_a)
        endpoint_b = __ProducerConsumer__(consumer_b, producer_b)
        return (endpoint_a, endpoint_b)
    else:
        if buffer_limit == None:
            queue = Queue()
        else:
            queue = Queue(buffer_limit)
        
        consumer = __QueueConsumer__(queue)
        producer = __QueueProducer__(queue)
        return (consumer, producer)

def EndPoint(consumer, producer):
    return __ProducerConsumer__(consumer, producer)

class __ProducerConsumer__:
    def __init__(self, consumer, producer):
        self.consumer = consumer
        self.producer = producer

    def push(self, element, block=True):
        self.consumer.push(element, block=block)
    
    def pop(self, block=True):
        return self.producer.pop(block=block)

    def full(self):
        return self.consumer.full()

    def close(self, block=True):
        self.producer.close(block=block)
        self.consumer.close(block=block)

class __ConnectionConsumer__:
    def __init__(self, connection, lock=None, buffer_limit=None):
        self.lock = lock
        self.connection = connection
        self.buffer = deque([])
        self.flushing = False
        self.thread = None
        self.buffer_limit = buffer_limit
    
    def push(self, element, block=True):
        if self.connection == None:
            raise Exception('ChannelError: Inbound Connection Closed')
        if self.buffer_limit != None and len(self.buffer) > self.buffer_limit:
            return False
        if block:
            if self.thread != None:
                self.thread.join()
            self.__push__(element)
            return True
        else:
            self.buffer.appendleft(element)
            if not self.flushing:
                self.thread = Thread(target=self.__flush__)
                self.thread.daemon = True
                self.thread.start()
            if block:
                self.thread.join()
            return True
        
    def full(self):
        if self.buffer_limit == None:
            return False
        else:
            return len(self.buffer) > self.buffer_limit
    
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
                element = self.buffer.pop()
                self.__push__(element)
            except IndexError:
                break
            except OSError as e:
                self.buffer.appendleft(element)
                break
        self.flushing = False
    
    def __push__(self, element):
        if self.lock != None:
            with self.lock:
                self.connection.send(element)
        else:
            self.connection.send(element)


class __ConnectionProducer__:
    def __init__(self, connection, lock):
        self.lock = lock
        self.connection = connection
    

    def pop(self, block=True):
        if self.connection == None:
            raise Exception('ChannelError: Outbound Connection Closed')
        
        # Note: 'None' indicates indefinite waiting, '0' indicates no waiting
        element = None
        if self.lock != None:
            with self.lock:
                if block:
                    try:
                        element = self.connection.recv()
                    except EOFError:
                        pass
                    except Exception as e:
                        print(e)
                elif self.connection.poll(0):
                    try:
                        element = self.connection.recv()
                    except EOFError:
                        pass
                    except Exception as e:
                        print(e)
        else:
            if block:
                try:
                    element = self.connection.recv()
                except EOFError:
                    pass
                except Exception as e:
                    print(e)
            elif self.connection.poll(0):
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



class __QueueConsumer__:
    def __init__(self, queue):
        self.queue = queue
    
    def push(self, element, block=True):
        try:
            self.queue.put(element, block)
            return True
        except Full:
            return False
        except BrokenPipeError as e:
            self.close()
            return False
        
    def full(self):
        return False
    
    def close(self, block=True):
        self.queue.close()
        if not block:
            self.queue.cancel_join_thread()

class __QueueProducer__:
    def __init__(self, queue):
        self.queue = queue
    
    def pop(self, block=True):
        try:
            element = self.queue.get(block)
            return element
        except Empty:
            return None
        except BrokenPipeError as e:
            self.close()
            return False

    def close(self, block=True):
        self.queue.close()
        if not block:
            self.queue.cancel_join_thread()
