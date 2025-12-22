import os
import asyncio
import socket
import subprocess

LLM_SERVER_PORT = 15311
LLM_SOCKET_PATH = "/tmp/unix_socket_async"

class UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, queue_record):
        self.queue_record = queue_record
    def connection_made(self, transport):
        self.transport = transport
    def datagram_received(self, data, addr):
        self.transport.sendto(str(self.queue_record[0]).encode(), addr)

# def get_llm_intensity_udp(scheduling_data_format):
#     try:
#         result = subprocess.run(["lsof", "-i:" + str(LLM_SERVER_PORT)], capture_output=True, text=True, check=True)
#     except subprocess.CalledProcessError:
#         return scheduling_data_format.default_data()
#     server_address = ('localhost', LLM_SERVER_PORT)
#     client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     client_socket.sendto("hi".encode(), server_address)
#     data = scheduling_data_format.read_data(client_socket)
#     client_socket.close()
#     return data

"""
    Read observation data from llm server using scheduling data format.
"""
def get_llm_intensity_unix(scheduling_data_format):
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client_socket.connect(LLM_SOCKET_PATH)
        client_socket.sendall(b"hi")
        data = scheduling_data_format.read_data(client_socket)
        client_socket.close()
    except:
        return scheduling_data_format.default_data()
    return data