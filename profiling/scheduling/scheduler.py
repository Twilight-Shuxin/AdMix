import time
from enum import IntEnum
from pydantic import BaseModel
from profiling.scheduling.monitor import llm_server_rpc

def pack_data_bytes(data_object : BaseModel):
    return data_object.json().encode("utf-8")

def get_data_length(conn):
    length_bytes = conn.recv(4)
    return int.from_bytes(length_bytes, "big")

"""
    Get data from a connection, where the data will be interpreted as a data_cls object.
    Expect the data_cls object to have a parse_raw class function to transform the received raw data into the actual object.
    
    chunked: whether to receive the data by chunks
"""
def conn_get_data(data_cls, conn, chunked = False):
    # Get data length
    length = get_data_length(conn)
    # Read data
    data = conn.recv(length) if not chunked else conn_get_data_chunked(conn, length)
    return data_cls.parse_raw(data.decode("utf-8"))

def conn_get_data_chunked(conn, data_length):
    buf = bytearray()
    while len(buf) < data_length:
        chunk = conn.recv(data_length - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)

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

    def start_serve(self):
        self.model_client.command_workers("sched_serve", [self.model_client.test_config])
        llm_server_rpc("ready")

        self.model_client.runtime_data.start_time_pointer = time.perf_counter()
        self.model_client.runtime_data.time_limit = self.model_client.runtime_data.start_time_pointer + self.model_client.test_config.test_time

    def serve(self, model_client):
        self.model_client = model_client
        self._serve()

class AIMDBaseScheduler(Scheduler):
    class Action(IntEnum):
        UP = 3
        FLAT = 2
        DOWN = 1
    
    def scheduler_init(self):
        profiled_data = self.model_client.read_profiled_data()
        max_batch_size = self.model_client.supported_batchsize[-1]
        for i in reversed(range(1, max_batch_size + 1)):
            if(profiled_data[str(i)] < self.time_interval):
                max_batch_size = i
                break
        self.max_batch_size = max_batch_size
        self.alpha = self.model_client.alpha
        self.beta = self.model_client.beta
    
    def step_up(self):
        if(self.batch_size == 0):
            self.batch_size += 1
            return
        if(self.batch_size < self.max_batch_size):
            self.batch_size += self.alpha if (self.batch_size != 1 or self.alpha == 1) else self.alpha - 1
            self.batch_size = min(self.max_batch_size, self.batch_size)
    
    def step_down(self):
        self.batch_size = self.batch_size // self.beta
    
    def execute_action(self, action):
        match action:
            case self.Action.UP:
                self.step_up()
            case self.Action.FLAT:
                if(self.batch_size == 0):
                    self.step_up()
            case self.Action.DOWN:  
                self.step_down()
