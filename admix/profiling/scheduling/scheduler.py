import time
from enum import IntEnum
from pydantic import BaseModel
from admix.profiling.scheduling.monitor import llm_server_rpc

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
        self.llm_slo_info = {k: v * 0.8 for k, v in llm_server_rpc("slo").items()} # {"first", "idle", "speed")
        llm_server_rpc("ready")

        self.model_client.runtime_data.start_time_pointer = time.perf_counter()
        self.model_client.runtime_data.time_limit = self.model_client.runtime_data.start_time_pointer + self.model_client.test_config.test_time

    def serve(self, model_client):
        self.model_client = model_client
        self._serve()
