import unittest

from lib.parallel.channel import Channel

class TestChannel(unittest.TestCase):

    def test_channel_flush(self):
        channel = Channel()
        channel.identify('both')
        for i in range(10):
            channel.push(i)
        output = [ channel.pop() for _i in range(10) ]
        channel.close()
        self.assertEqual(output, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

if __name__ == '__main__':
    unittest.main()
