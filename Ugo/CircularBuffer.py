import random as rd

class CircularBuffer:
    def __init__(self, size):
        self.buffer = [0]*size
        self.start_index = 0
        self.size = size

    def push(self, element):
        self.buffer[self.start_index] = element
        self.start_index = (self.start_index + 1) % self.size

    def sample(self, number):
        return rd.choices(self.buffer, k=number)