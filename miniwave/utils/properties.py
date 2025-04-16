class Properties:

    def __init__(
            self,
            block_size_1: int,
            block_size_2: int,
            block_size_3: int
    ):

        self.block_size_1 = block_size_1
        self.block_size_2 = block_size_2
        self.block_size_3 = block_size_3
        self._space_order = None
        self._dtype = None

    @property
    def block3(self):
        return (self.block_size_1, self.block_size_2, self.block_size_3)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value: str):
        self._dtype = value

    @property
    def space_order(self):
        return self._space_order

    @space_order.setter
    def space_order(self, value: str):
        self._space_order = value

    @property
    def stencil_radius(self):
        if self.space_order is not None:
            return self.space_order // 2

        return None
