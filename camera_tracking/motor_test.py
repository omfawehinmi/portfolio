import machine
from time import sleep
from configuration import disable_motors
import asyncio


def run_motor1():
    # Define pin numbers
    step_pins = [18, 19, 20, 21]
    ena_pin = 22  # Replace with the actual pin number you are using for the ENA pin

    # Initialize pins
    step_pins = [machine.Pin(pin, machine.Pin.OUT) for pin in step_pins]
    ena_pin = machine.Pin(ena_pin, machine.Pin.OUT)

    # Assuming a full revolution requires 200 steps (adjust based on your motor specifications)
    steps_per_revolution = 200

    # Calculate the number of steps for the desired angle (e.g., 10 degrees)
    desired_angle = 180
    steps_for_desired_angle = int(steps_per_revolution * (desired_angle / 360.0))

    full_step_sequence = [
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0]
    ]

    reverse_full_step_sequence = [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 1]
    ]

    # Enable the motor driver
    ena_pin.value(0)  # 1 for enabling, 0 for disabling

    # Rotate forward
    for _ in range(steps_for_desired_angle):
        for step in full_step_sequence:
            for i in range(len(step_pins)):
                step_pins[i].value(step[i])
            sleep(0.001)  # Reduced sleep time for faster movement

    # Rotate backward
    for _ in range(steps_for_desired_angle):
        for step in reverse_full_step_sequence:
            for i in range(len(step_pins)):
                step_pins[i].value(step[i])
            sleep(0.001)  # Reduced sleep time for faster movement

    # Disable the motor driver when the program ends (optional)
    disable_motors()
    
def run_motor2():
    # Define pin numbers
    step_pins = [2, 3, 4, 5]
    ena_pin = 6  # Replace with the actual pin number you are using for the ENA pin

    # Initialize pins
    step_pins = [machine.Pin(pin, machine.Pin.OUT) for pin in step_pins]
    ena_pin = machine.Pin(ena_pin, machine.Pin.OUT)

    # Assuming a full revolution requires 200 steps (adjust based on your motor specifications)
    steps_per_revolution = 200

    # Calculate the number of steps for the desired angle (e.g., 10 degrees)
    desired_angle = 180
    steps_for_desired_angle = int(steps_per_revolution * (desired_angle / 360.0))

    full_step_sequence = [
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0]
    ]

    reverse_full_step_sequence = [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 1]
    ]

    # Enable the motor driver
    ena_pin.value(0)  # 1 for enabling, 0 for disabling

    # Rotate forward
    for _ in range(steps_for_desired_angle):
        for step in full_step_sequence:
            for i in range(len(step_pins)):
                step_pins[i].value(step[i])
            sleep(0.001)  # Reduced sleep time for faster movement

    # Rotate backward
    for _ in range(steps_for_desired_angle):
        for step in reverse_full_step_sequence:
            for i in range(len(step_pins)):
                step_pins[i].value(step[i])
            sleep(0.001)  # Reduced sleep time for faster movement

    # Disable the motor driver when the program ends (optional)
    disable_motors()
    
