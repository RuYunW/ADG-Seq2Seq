class Node():
    def __init__(self, onehot, front_node, next_node):
        self.onehot = onehot
        self.front_node = front_node
        self.next_node = next_node

    def get_onehot(self):
        return self.onehot

    