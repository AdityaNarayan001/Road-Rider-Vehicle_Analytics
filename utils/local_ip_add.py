import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))  # 8.8.8.8 is a Google DNS server
        local_ip = s.getsockname()[0]
    except Exception as e:
        local_ip = f"Error: {e}"
    finally:
        s.close()
    return local_ip

print(f"Local IP Address: {get_local_ip()}")

