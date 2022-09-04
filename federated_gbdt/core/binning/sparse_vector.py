# sparse vector ------------------------------------------------------
class SparseVector:
    """
    sparse vector: dict, record (indices, data) kv tuples
    shape: the real feature shape of data
    """
    def __init__(self, indices=None, fn=None, data=None, shape=0):
        self.sparse_vec = dict(zip(indices, data))
        self.feature_name = fn
        self.shape = shape

    def get_all_data(self):
        for idx, data in self.sparse_vec.items():
            yield idx, data

    def get_shape(self):
        return self.shape
