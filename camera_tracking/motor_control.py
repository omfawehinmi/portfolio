import machine
import uasyncio as asyncio
from configuration import disable_motors
import configuration as cf
from timeout import Queue, TimeoutError
import shared_variables as sv

async def run_motor(direct, ena_pin, step_pins, steps, slp_duration):
    ena_pin.value(1)

    try:
        if not sv.MOTORS_RUNNING:
            KeyboardInterrupt
            
        print(f"Steps: {steps}")
        for _ in range(abs(steps)):
            for step in direct:
                for i in range(len(step_pins)):
                    step_pins[i].value(step[i])
                await asyncio.sleep(slp_duration)
            
            if _ > cf.decel_step * abs(steps):
                slp_duration *= cf.decel 
                
                
            if ena_pin == cf.ena_pin_x:
                if direct == cf.forward:
                    sv.adjusted_x_coord += -1
                else: 
                    sv.adjusted_x_coord += 1
                
            elif ena_pin == cf.ena_pin_y:
                if direct == cf.forward:
                    sv.adjusted_y_coord += -1
                else:
                    sv.adjusted_y_coord += 1
        
        disable_motors()

    except KeyboardInterrupt:
        disable_motors()
        
async def run_motor_no_tracking(direct, ena_pin, step_pins, steps, slp_duration):
    ena_pin.value(1)

    try:
        print(f"Steps: {steps}")
        for _ in range(abs(steps)):
            for step in direct:
                for i in range(len(step_pins)):
                    step_pins[i].value(step[i])
                await asyncio.sleep(slp_duration) 
            
    except KeyboardInterrupt:
        disable_motors()

        
async def move_motors_no_tracking(x_coord, y_coord):
    steps_x = abs(x_coord)
    slp_duration_x = 0.0001
    direct_x = cf.forward if x_coord <= 0 else cf.reverse
    
    #Normalize coordinate
    steps_y = abs(y_coord)
    slp_duration_y = 0.0001
    direct_y = cf.forward if y_coord <= 0 else cf.reverse

    try:
        task_x = asyncio.create_task(run_motor_no_tracking(direct_x, cf.ena_pin_x, cf.step_pins_x, steps_x, slp_duration_x))
        task_y = asyncio.create_task(run_motor_no_tracking(direct_y, cf.ena_pin_y, cf.step_pins_y, steps_y, slp_duration_y))
        await asyncio.gather(task_x, task_y)
        
        disable_motors()

    except KeyboardInterrupt:
        disable_motors()


async def move_motors(x_coord, y_coord):
    x_coord = x_coord * cf.scale_x
    full_turn_steps_x = 1800
    max_x_coord = cf.frame_x
    steps_x = int((abs(x_coord) / max_x_coord) * full_turn_steps_x)
    slp_duration_x = 0.0001
    direct_x = cf.forward if x_coord <= 0 else cf.reverse
    
    y_coord = y_coord * cf.scale_y
    full_turn_steps_y = 1800
    max_y_coord = cf.frame_y
    steps_y = int((abs(y_coord) / max_y_coord) * full_turn_steps_y)
    slp_duration_y = 0.0001
    direct_y = cf.forward if y_coord <= 0 else cf.reverse
    

    try:
        task_x = asyncio.create_task(run_motor(direct_x, cf.ena_pin_x, cf.step_pins_x, steps_x, slp_duration_x))
        task_y = asyncio.create_task(run_motor(direct_y, cf.ena_pin_y, cf.step_pins_y, steps_y, slp_duration_y))
        await asyncio.gather(task_x, task_y)
        
        disable_motors()

    except KeyboardInterrupt:
        disable_motors()

async def scan_room():
    full_turn_steps_x = 800
    desired_angle_x = 180
    slp_duration_x = 0.01
    steps_x = int((desired_angle_x / 360.0) * full_turn_steps_x)

    full_turn_steps_y = 1600
    desired_angle_y = 20
    slp_duration_y = 0.003
    steps_y = int((desired_angle_y / 360.0) * full_turn_steps_y)
    
    
    direct_x = cf.forward
    direct_y = cf.forward
    try:
        while sv.MOTORS_RUNNING:
            if not sv.MOTORS_RUNNING:
                    break
            if direct_x == cf.forward:
                task_x = asyncio.create_task(run_motor(direct_x, cf.ena_pin_x, cf.step_pins_x, steps_x, slp_duration_x))
                direct_x = cf.reverse
            else:
                task_x = asyncio.create_task(run_motor(direct_x, cf.ena_pin_x, cf.step_pins_x, steps_x, slp_duration_x))
                direct_x = cf.forward
            
            for _ in range(13):
                if not sv.MOTORS_RUNNING:
                    break
                if direct_y == cf.forward:
                    task_y = asyncio.create_task(run_motor(direct_y, cf.ena_pin_y, cf.step_pins_y, steps_y, slp_duration_y))
                    direct_y = cf.reverse
                else:
                    task_y = asyncio.create_task(run_motor(direct_y, cf.ena_pin_y, cf.step_pins_y, steps_y, slp_duration_y))
                    direct_y = cf.forward
                await task_y

            await task_x
            
    except KeyboardInterrupt:
        disable_motors()
        
async def process_coordinates():
    while True:
        if sv.write_event_triggered:
            sv.MOTORS_RUNNING = False
            try:
                x, y = await sv.coordinate_queue.dequeue(timeout=10)
                print(f"X={x}, Y={y}")
                
                await move_motors(x, y)
                
            except TimeoutError:
                print("TimeoutError: No coordinates available")
                try:
                    asyncio.create_task(stop_motors()) 
                    
                except Exception as e:
                    print("An error occurred during reset or scan:", e)
                    asyncio.create_task(stop_motors()) 
                    
        await asyncio.sleep(0)

async def stop_motors():
    sv.MOTORS_RUNNING = False
    asyncio.run(reset_motors()) 
    print("Motors Stopped")
    
async def reset_motors():
    x_coord = sv.adjusted_x_coord
    y_coord = sv.adjusted_y_coord
    
    await move_motors_no_tracking(-1 * x_coord, -1 * y_coord)
    
    sv.adjusted_x_coord = 0
    sv.adjusted_y_coord = 0
    
    return x_coord, y_coord

