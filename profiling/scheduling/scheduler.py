import time
from enum import IntEnum
from pydantic import BaseModel
from profiling.scheduling.monitor import get_llm_info_unix

def pack_data_bytes(data_object : BaseModel):
    return data_object.json().encode("utf-8")

def get_data_length(conn):
    length_bytes = conn.recv(4)
    return int.from_bytes(length_bytes, "big")

"""
    Get data from a connection, where the data will be interpreted as a data_cls object.
    Expect the data_cls object to have a parse_raw class function to transform the received raw data into the actual object.
"""
def conn_get_data(data_cls, conn):
    # Get data length
    length = get_data_length(conn)
    # Read data
    data = conn.recv(length)
    return data_cls.parse_raw(data.decode("utf-8"))

class SchedulingDataFormat():
    def default_data():
        pass
    
    def pack_data(llm_worker):
        pass
    
    def read_data(client_socket):
        pass

class Scheduler():
    profiling_executor = None
    monitor = None

    def serve(self, model_worker):
        self.model_worker = model_worker

        self._serve()
        # Be informed that the llm worker is ready to accept queries through unix server
        self.model_worker.double_barrier.wait()

        # Update test_config according to llm worker, and re-initialize test environment data
        self.model_worker.test_config.test_time = get_llm_info_unix()
        self.model_worker.init_testenv_data()
        
        self._serve()

class AIMDBaseScheduler(Scheduler):
    class Action(IntEnum):
        UP = 3
        FLAT = 2
        DOWN = 1
    
    def step_up(self):
        if(self.batch_size == 0):
            self.batch_size += 1
            return
        if(self.batch_size < self.max_batch_size):
            self.batch_size += self.alpha if (self.batch_size != 1 or self.alpha == 1) else self.alpha - 1
            self.batch_size = min(self.max_batch_size, self.batch_size)
    
    def step_down(self):
        self.batch_size = self.batch_size // self.beta
    
    @classmethod
    def execute_action(cls, action):
        match action:
            case self.Action.UP:
                self.step_up()
            case self.Action.FLAT:
                if(self.batch_size == 0):
                    self.step_up()
            case self.Action.DOWN:  
                self.step_down()
