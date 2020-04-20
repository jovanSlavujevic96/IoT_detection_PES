import socket
from threading import Thread
import threading
import time
import RPi.GPIO as GPIO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--IP', 
        type=str,
        help='set the IP address of the rPi (server device)')

args = parser.parse_args()
host = str(args.IP)
port = 12345

print('IP address: ' + str(args.IP) )
print('port:       ' + str(port) )

GPIO.setmode(GPIO.BCM)
GPIO.setup(17,GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def listener(client, address):
    print ("\nAccepted connection from: ", address,'\n')
    global sensor_val
    with clients_lock:
        clients.add(client)
    try:    
        while True:
            data = client.recv(1024)
            if int(data) == 0:
                catch = sensor_val
                print('\nSENDING: ', str(catch),'\n')
                char = str(int(catch))
                client.send(bytes(char,encoding='utf8'))
    except ValueError:
        print("\nClient Disconnected\n")
    
clients = set()
clients_lock = threading.Lock()

s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((host,port))
s.listen(3)
th = []

def GPIO_read():
    global sensor_val
    global disconnect
    disconnect = False
    while True:
        sensor_val = bool(GPIO.input(17) )
        sensor_val = not sensor_val
        print('alcohol detected: ', str(sensor_val))
        time.sleep(0.1)

try:
    th.append(Thread(target=GPIO_read).start())
    while True:
        print ("\nServer is listening for connections...\n")
        client, address = s.accept()

        th.append(Thread(target=listener, args = (client,address)).start())
except KeyboardInterrupt:
    s.close()
    print("\nSucc closed server\n")
    exit() 
