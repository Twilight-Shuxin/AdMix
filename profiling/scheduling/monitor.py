import os
import time
import socket
import asyncio
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

def llm_server_rpc(action):
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client_socket.connect(LLM_SERVER_SOCKET_PATH)
    client_socket.sendall(action.encode("utf-8"))
    data = None
    match action:
        case "test_time":
            data = int.from_bytes(client_socket.recv(4), "big")
        case "ready":
            pass
        case "tag":
            data = client_socket.recv(1024).decode("utf-8")
        case "slo":
            import json
            length = int.from_bytes(client_socket.recv(4), "big")
            data = json.loads(client_socket.recv(length))
    client_socket.close()
    return data

"""
    Communicates with llm server. 
    FIXME: Retries forever until information is correctly received. 
"""
def llm_server_rpc_retry(action):
    retry_interval = 0.5
    while True:
        try:
            data = llm_server_rpc(action)
            break
        except Exception as e:
            time.sleep(retry_interval)
    return data