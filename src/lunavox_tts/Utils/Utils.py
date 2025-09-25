from collections import OrderedDict
import queue


class LRUCacheDict(OrderedDict):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)  # 访问后移到末尾
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.capacity:
            self.popitem(last=False)  # 删除最旧的（第一个）


def clear_queue(q: queue.Queue) -> None:
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break
