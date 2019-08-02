import multiprocessing

class NonDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class UnsafePool(multiprocessing.pool.Pool):
    Process = NonDaemonProcess

    def __del__(self):
        try:
            self.close()
            super(UnsafePool, self).__del__()
        except:
            pass
