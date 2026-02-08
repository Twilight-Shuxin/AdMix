import os
import asyncio
import socket
import subprocess

LLM_SOCKET_PATH = "/tmp/unix_socket_async"

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
    except Exception as e:
        return scheduling_data_format.default_data()
    return data