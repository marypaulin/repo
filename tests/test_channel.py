import unittest
from lib.parallel.actor import Actor
from lib.parallel.channel import Channel

class TestChannel(unittest.TestCase):

    def test_locking_channel_flush(self):
        consumer, producer = Channel(read_lock=True, write_lock=True)

        def produce(id, services, termination):
            (consumer,) = services
            for i in range(10):
                consumer.push(i, block=True)
            consumer.close()

        process = Actor(0, (consumer,), produce, actor_type='process')
        process.start()

        output = [producer.pop(block=True) for _i in range(10)]
        producer.close()

        self.assertEqual(output, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_non_locking_channel_flush(self):
        consumer, producer = Channel(read_lock=False, write_lock=False)

        def produce(id, services, termination):
            (consumer,) = services
            for i in range(10):
                consumer.push(i, block=True)
            consumer.close()

        
        process = Actor(0, (consumer,), produce, actor_type='process')
        process.start()

        output = [producer.pop(block=True) for _i in range(10)]
        producer.close()

        self.assertEqual(output, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

if __name__ == '__main__':
    unittest.main()
