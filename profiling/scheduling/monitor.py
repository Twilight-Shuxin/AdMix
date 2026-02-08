import os
import asyncio
import socket
import subprocess

LLM_INTENSITY_SOCKET_PATH = "/tmp/llm_intensity_socket"
LLM_SERVER_SOCKET_PATH = "/tmp/llm_server_socket"

"""
    Read observation data from llm server using scheduling data format.
"""
def get_llm_intensity_unix(scheduling_data_format):
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client_socket.connect(LLM_INTENSITY_SOCKET_PATH)
        client_socket.sendall(b"hi")
        data = scheduling_data_format.read_data(client_socket)
        client_socket.close()
    except Exception as e:
        return scheduling_data_format.default_data()
    return data

"""
    Read requested information from llm server. 
    FIXME: Current implementation gets only the test_time of llm server, 
"""
def get_llm_info_unix():
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client_socket.connect(LLM_SERVER_SOCKET_PATH)
        client_socket.sendall(b"test_time")
        data = int.from_bytes(client_socket.recv(4), "big")
        client_socket.close()
    except Exception as e:
        return 
    return data