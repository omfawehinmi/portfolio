from timeout import Queue

MOTORS_RUNNING = True

steps_updated_x = False
steps_updated_y = False

adjusted_x_coord = 0
adjusted_y_coord = 0

coordinate_queue = Queue()

write_event_triggered = False