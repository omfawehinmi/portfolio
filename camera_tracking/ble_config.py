import asyncio
from bleak import BleakScanner, BleakClient
'''_________________________________________________________________________________________________________________________________________________________________________'''
target_device_name = "micro"
'''_________________________________________________________________________________________________________________________________________________________________________'''
async def search_for_device(target_device_name: str):
    while True:
        devices = await BleakScanner.discover()
        target_device = next((dev for dev in devices if dev.name == target_device_name), None)
        if target_device:
            print(f"Found {target_device_name} with address {target_device.address}")
            return target_device
        else:
            print(f"{target_device_name} not found in the scanned devices. Retrying...")

'''_________________________________________________________________________________________________________________________________________________________________________'''
async def send_coordinates(x, y, client, frame_width, frame_height):
    try:
        if client.is_connected:
            center_x, center_y = frame_width / 2 - 75, frame_height / 2 -100

            x_distance = (x - center_x) / frame_width
            y_distance = (y - center_y) / frame_height

            # Format X and Y coordinates with 6 decimal places
            x_str = "{:.6f}".format(x_distance)

            if y == 0:
                y_str = "{:.6f}".format(0)
            else:
                y_str = "{:.6f}".format(y_distance)

            # Format X and Y coordinates into a single string with a space delimiter
            coordinates_str = f"{x_str} {y_str}\n"

            # Encode the coordinates string to bytes
            data_bytes = coordinates_str.encode('utf-8')

            # Write the bytes to the characteristic with handle 0x0008
            await client.write_gatt_char(0x0008, data_bytes)

            print(f"Coordinates sent successfully over Bluetooth: {coordinates_str}")
        else:
            print("Error: Client is not connected to the Bluetooth device")
    except Exception as e:
        print(f"Error sending coordinates over Bluetooth: {e}")


