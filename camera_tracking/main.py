from motor_test import run_motor1, run_motor2
from ble_detection import bluetooth_detection
from motor_control import stop_motors, reset_motors, process_coordinates
import uasyncio as asyncio

async def main():
    run_motor1()
    run_motor2()
    
    task1 = asyncio.create_task(bluetooth_detection())
    task2 = asyncio.create_task(process_coordinates())
    
    await asyncio.gather(task1, task2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
        
    except KeyboardInterrupt:
        asyncio.run(stop_motors())