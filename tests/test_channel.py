import unittest
from multiprocessing import Process
from lib.parallel.channel import Channel

class TestChannel(unittest.TestCase):

    def test_channel_flush(self):
        consumer, producer = Channel(consumers=2, producers=2)

        def produce(consumer):
            for i in range(10):
                consumer.push(i, block=False)
            consumer.close()

        process = Process(target=produce, args=(consumer,))
        process.start()
        
        output = [ producer.pop(block=True) for _i in range(10) ]
        producer.close()

        self.assertEqual(output, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

if __name__ == '__main__':
    unittest.main()
