import unittest
import analysis as aa
import numpy as np

class TestFifoTradeQueue(unittest.TestCase):
    def test_fifo_trade_queue(self):
        # follow the example in code in analysis.py
        trades = [+100, +50, -20, -50, -80]
        timestamps = [0, 1, 2, 3, 4]

        reconstructed_position = np.zeros(5)

        queue = aa.FifoTradeQueue(10)
        for timestamp, trade in zip(timestamps, trades):
            x = queue.push(timestamp, trade)
            for trade, t1, t2 in x:
                reconstructed_position[t1:t2] += trade

        self.assertTrue(
            np.allclose(reconstructed_position, np.cumsum(trades)))


if __name__ == '__main__':
    unittest.main()
