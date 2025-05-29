import os
import asyncio
import socket
import subprocess
import struct
from profiling.scheduling.scheduler import default_AIMDScheduler_data

LLM_SERVER_PORT = 15311
LLM_SOCKET_PATH = "/tmp/unix_socket_async"

class UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, queue_record):
        self.queue_record = queue_record
    def connection_made(self, transport):
        self.transport = transport
    def datagram_received(self, data, addr):
        self.transport.sendto(str(self.queue_record[0]).encode(), addr)

def get_llm_intensity_udp(default_data):
    try:
        result = subprocess.run(["lsof", "-i:" + str(LLM_SERVER_PORT)], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return default_data
    server_address = ('localhost', LLM_SERVER_PORT)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.sendto("hi".encode(), server_address)
    data, _ = client_socket.recvfrom(32)
    client_socket.close()
    return data.decode()

def get_llm_intensity_unix(default_data):
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client_socket.connect(LLM_SOCKET_PATH)
    except:
        return default_data
    client_socket.sendall(b"hi")
    data = client_socket.recv(32)
    client_socket.close()
    return struct.unpack("iii", data)