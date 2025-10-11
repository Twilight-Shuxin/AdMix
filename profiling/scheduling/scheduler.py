import time
from pydantic import BaseModel

def pack_data_bytes(data_object : BaseModel):
    return data_object.json().encode("utf-8")

def unpack_data_bytes(data_cls, conn, length):
    data = conn.recv(length)
    return data_cls.parse_raw(data.decode("utf-8"))

def get_data_length(conn):
    length_bytes = conn.recv(4)
    length = int.from_bytes(length_bytes, "big")
    return length

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

    def serve(self, model_worker, test_time, rec):
        pass