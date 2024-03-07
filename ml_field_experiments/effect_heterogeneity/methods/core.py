class Test(object):
    def __init__(self, verbose=False, w_hat=None) -> None:
        self.verbose = verbose
        self.w_hat = w_hat

    def run(self, x, z, y, tau=None):
        raise NotImplementedError("Test must implement run method")
