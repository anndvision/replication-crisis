class Test(object):
    def __init__(self, w_hat=None, verbose=False) -> None:
        self.w_hat = w_hat
        self.verbose = verbose

    def run(self, x, z, y):
        raise NotImplementedError("Test must implement run method")
