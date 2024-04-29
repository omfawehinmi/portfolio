import uasyncio as asyncio
import bluetooth
from machine import Pin
from ble_advertising import advertising_payload
from motor_control import move_motors, scan_room, stop_motors, reset_motors
from micropython import const
import time
from motor_test import run_motor1, run_motor2
import shared_variables as sv
import configuration as cf

# Define constants for IRQ events and flags
_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATTS_WRITE = const(3)

_FLAG_READ = const(0x0002)
_FLAG_WRITE_NO_RESPONSE = const(0x0004)
_FLAG_WRITE = const(0x0008)
_FLAG_NOTIFY = const(0x0010)

# Define UUIDs for UART service and characteristics
_UART_UUID = bluetooth.UUID("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
_UART_TX = (
    bluetooth.UUID("6E400003-B5A3-F393-E0A9-E50E24DCCA9E"),
    _FLAG_READ | _FLAG_WRITE | _FLAG_NOTIFY,
)
_UART_RX = (
    bluetooth.UUID("6E400002-B5A3-F393-E0A9-E50E24DCCA9E"),
    _FLAG_WRITE | _FLAG_NOTIFY,
)

_UART_SERVICE = (
    _UART_UUID,
    (_UART_TX, _UART_RX),
)

# Class for BLE peripheral
class BLESimplePeripheral:
    def __init__(self, ble, name="micro"):
        self._ble = ble
        self._ble.active(True)
        self._ble.irq(self._irq)
        ((self._handle_tx, self._handle_rx),) = self._ble.gatts_register_services((_UART_SERVICE,))
        self._connections = set()
        self._write_callback = None
        self._payload = advertising_payload(name=name, services=[_UART_UUID])
        self._advertise()

    def _irq(self, event, data):
        if event == _IRQ_CENTRAL_CONNECT:
            print("Connected to detection device")
            conn_handle, _, _ = data
            self._connections.add(conn_handle)
            
        elif event == _IRQ_CENTRAL_DISCONNECT:
            print("Disconnected from detection device")
            conn_handle, _, _ = data
            self._connections.remove(conn_handle)
            
            stop_motors()

            sv.write_event_triggered = False
            self._advertise()
            
            
        elif event == _IRQ_GATTS_WRITE:
            sv.write_event_triggered = True
            
            conn_handle, value_handle = data
            value = self._ble.gatts_read(value_handle)
            decoded_data = value.decode().strip()

            data_values = decoded_data.split(" ")
            
            try:
                x, y = map(float, data_values)
                
                if x < 0:
                    x_norm = x * cf.frame_x + cf.center_x
                    if abs(x_norm) < 0.025:
                        x = 0.00
                else:
                    if abs(x) < 0.025:
                       x = 0.00
                       
                       
                if y < 0:
                    y_norm = y * cf.frame_y + cf.center_y
                
                    if abs(y_norm) < 0.025:
                        y = 0.00
                        
                else:
                    if abs(y) < 0.025:
                       y = 0.00
                       
                asyncio.create_task(sv.coordinate_queue.enqueue((x, y)))               
                
            except (ValueError, IndexError) as e:
                print(f"Error processing data: {e}")

            if value_handle == self._handle_rx and self._write_callback:
                self._write_callback(value)

    def send(self, data):
        for conn_handle in self._connections:
            self._ble.gatts_notify(conn_handle, self._handle_tx, data)

    def is_connected(self):
        return len(self._connections) > 0

    def _advertise(self, interval_us=500000):
        print("Starting advertising")
        self._ble.gap_advertise(interval_us, adv_data=self._payload)
        try:
            asyncio.create_task(scan_room())
        
        except KeyboardInterrupt:
            stop_motors()
            
    def on_write(self, callback):
        self._write_callback = callback
        
        
async def connection():
    ble = bluetooth.BLE()
    p = BLESimplePeripheral(ble)

    i = 0
    while True:
        if p.is_connected():
            for _ in range(3):
                data = str(i) + "_"
                p.send(data)
                i += 1
        await asyncio.sleep_ms(100)

# Define the GATT handle for RX characteristic
RX_CHAR_HANDLE = 36 

async def bluetooth_detection():
    task_connection = asyncio.create_task(connection())
    try:
        while True:
            await asyncio.sleep_ms(0)

    except Exception as e:
        print("An error occurred in bluetooth_detection:", e)
