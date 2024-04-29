import machine
import asyncio

ena_pin_x = machine.Pin(22, machine.Pin.OUT)
step_pins1 = [18, 19, 20, 21]
step_pins_x = [machine.Pin(pin, machine.Pin.OUT) for pin in step_pins1]

ena_pin_y = machine.Pin(6, machine.Pin.OUT)
step_pins2 = [2, 3, 4, 5]
step_pins_y = [machine.Pin(pin, machine.Pin.OUT) for pin in step_pins2]


decel_step = .80
decel = .10

scale_x = 40
scale_y = 30

frame_x = 640
frame_y = 480
center_x, center_y = frame_x / 2 - 75, frame_y / 2 - 100



def disable_motors():
    global ena_pin_x, ena_pin_y, step_pins_x, step_pins_y

    pins = step_pins_x + step_pins_y + [ena_pin_x] + [ena_pin_y]

    for pin in pins:
        pin.value(0)
    
    
reverse = [
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 1, 0]
]

forward = [
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 0, 1]
]
        


            



