import socket
import time
import json

SHUXIN_LLM_SOCKET_PATH = "/tmp/unix_socket_async"

def get_data_length(conn):
    length_bytes = conn.recv(4)
    length = int.from_bytes(length_bytes, "big")
    return length

def main():
    for i in range(100):
        client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # try:
        #     client_socket.connect(SHUXIN_LLM_SOCKET_PATH)
        #     client_socket.sendall(b"hi")
        #     length = get_data_length(client_socket)
        #     data = conn.recv(length)
        #     data = json.loads(data.decode("utf-8"))
        #     client_socket.close()
        # except:
        #     data = None
        client_socket.connect(SHUXIN_LLM_SOCKET_PATH)
        client_socket.sendall(b"hi")
        length = get_data_length(client_socket)
        data = client_socket.recv(length)
        data = json.loads(data.decode("utf-8"))
        client_socket.close()
        print(data)
        time.sleep(0.1)

if __name__ == "__main__":
    main()