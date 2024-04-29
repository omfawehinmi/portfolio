# Camer Tracking
## Purpose
This project explores the intersection of machine learning and physical infrastructure. It involves a comprehensive system that leverages various components such as Bluetooth Low Energy (BLE), motor control, webcam configuration, and more to demonstrate how machine learning can be applied to control and optimize physical systems.

Components Summary
Configuration and Setup
Configuration Files: Scripts like configuration.py, gpu_config.py, and webcam_config.py handle the setup and configuration of the system's various hardware and software parameters, such as GPU settings and webcam resolutions.
BLE Configuration: The ble_config.py script sets up parameters and utilities for Bluetooth operations, crucial for communication within the infrastructure.
Main Operational Scripts
Main Coordinator (main_coord.py): Acts as the central script that coordinates various tasks and operations, processing inputs and controlling outputs based on system configurations and real-time data.
Robotics Control (main_robo.py): Manages robotics-specific operations, integrating sensors and actuators with decision-making processes driven by machine learning insights.
Device Communication and Control
BLE Advertising and Detection: Scripts ble_advertising.py and ble_detection.py manage the advertisement and detection of BLE devices, facilitating the wireless communication essential for remote operations.
Motor Control (motor_control.py): Controls the motors based on commands derived from machine learning algorithms, adjusting movements to optimize task execution.
Utilities and Shared Resources
Shared Variables (shared_variables.py): Manages shared resources and variables that are used across different scripts, ensuring consistent and synchronized operations.
Project Rationale
This project was driven by a fascination with how machine learning can enhance and interact with physical infrastructures. By integrating intelligent algorithms with physical components, the project demonstrates potential improvements in automation, efficiency, and functionality in real-world applications.
