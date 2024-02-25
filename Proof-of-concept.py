import tkinter as tk
from evdev import UInput, ecodes as e, AbsInfo
import threading
import time
import random
import subprocess
import re
import cv2
import numpy as np
from PIL import Image
import io
from torchvision import transforms
import torch
from torch import nn
from torch.nn import functional as F

class FoxAI(nn.Module):
    def __init__(self):
        super(FoxAI, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(24*24*64, 512)  # Adjusting to the correct input size
        self.fc2 = nn.Linear(512, 12)  # 12 possible actions on SNES controller

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 24*24*64)  # Reshape the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class VirtualController:
    def __init__(self):
        self.capabilities = {
            e.EV_KEY: [
                e.BTN_EAST,  # A
                e.BTN_SOUTH, # B
                e.BTN_NORTH, # X
                e.BTN_WEST,  # Y
                e.BTN_TL,    # L
                e.BTN_TR,    # R
                e.BTN_SELECT,# SELECT
                e.BTN_START, # START
            ],
            e.EV_ABS: [
                (e.ABS_HAT0X, AbsInfo(0, -1, 1, 0, 0, 0)),  # Adjusted to include 'flat' and 'resolution'
                (e.ABS_HAT0Y, AbsInfo(0, -1, 1, 0, 0, 0)),  # Adjusted to include 'flat' and
            ]
        }
        self.device = UInput(events=self.capabilities, name="AI_Conquers_Ze_Galaxy")
        self.button_states = {button: False for button in self.capabilities[e.EV_KEY]}  # Track button states
        self.abs_states = {abs_event: 0 for abs_event, _ in self.capabilities[e.EV_ABS]}  # Track ABS states
        self.lock = threading.Lock()  # To ensure thread-safe updates to states

    def press_abs_conf(self, abs_event, value):
        self.device.write(e.EV_ABS, abs_event, value)
        self.device.syn()
        time.sleep(0.1)  # Mimic a brief press
        # Reset to neutral position
        self.device.write(e.EV_ABS, abs_event, 0)
        self.device.syn()

    def press_button_conf(self, button):
        self.device.write(e.EV_KEY, button, 1)  # Press the button
        self.device.syn()
        time.sleep(0.1)  # Hold the button briefly
        self.device.write(e.EV_KEY, button, 0)  # Release the button
        self.device.syn()
        
    def press_button(self, button, duration=0.1):
        with self.lock:
            if not self.button_states[button]:  # Button is not pressed
                self.device.write(e.EV_KEY, button, 1)  # Press the button
                self.button_states[button] = True
            else:  # Button is already pressed, release it
                self.device.write(e.EV_KEY, button, 0)
                self.button_states[button] = False
            self.device.syn()
            if duration > 0:  # Mimic a brief press
                self.press_button_conf(button)

    def press_abs(self, abs_event, value, reset_after=0.1):
        with self.lock:
            self.device.write(e.EV_ABS, abs_event, value)
            self.abs_states[abs_event] = value
            self.device.syn()
            if reset_after > 0:
                time.sleep(reset_after)
                # Reset to neutral position if specified
                self.device.write(e.EV_ABS, abs_event, 0)
                self.abs_states[abs_event] = 0
                self.device.syn()

    def start_auto_conf(self):
        self.press_button_conf(e.BTN_EAST) # Initializes the controller.
        def auto_press():
            # D-pad sequence
            directions = [
                (e.ABS_HAT0Y, -1),  # Up
                (e.ABS_HAT0Y, 1),   # Down
                (e.ABS_HAT0X, -1),  # Left
                (e.ABS_HAT0X, 1),   # Right
            ]
            # Button sequence
            buttons = [
                e.BTN_EAST,  # A
                e.BTN_SOUTH, # B
                e.BTN_NORTH, # X
                e.BTN_WEST,  # Y
                e.BTN_TL,    # L
                e.BTN_TR,    # R
                e.BTN_SELECT,# SELECT
                e.BTN_START, # START
            ]
            for abs_event, value in directions:
                time.sleep(10)
                self.press_abs_conf(abs_event, value)
                
            for button in buttons:
                time.sleep(10)
                self.press_button_conf(button)
        thread = threading.Thread(target=auto_press)
        thread.daemon = True
        thread.start()

    def execute_action(self, action):
        def action_to_controller(action):
            """
            Executes an action based on the prediction from FoxAI.
    
            :param action: An integer representing the predicted action.
            """
            action_methods = {
                0: lambda: self.press_button(e.BTN_EAST),    # Press A
                1: lambda: self.press_button(e.BTN_SOUTH),   # Press B
                2: lambda: self.press_button(e.BTN_NORTH),   # Press X
                3: lambda: self.press_button(e.BTN_WEST),    # Press Y
                4: lambda: self.press_button(e.BTN_TL),      # Press L
                5: lambda: self.press_button(e.BTN_TR),      # Press R
                6: lambda: self.press_abs(e.ABS_HAT0X, -1),  # Example: Move Left
                7: lambda: self.press_abs(e.ABS_HAT0X, 1),   # Example: Move Right
                8: lambda: self.press_abs(e.ABS_HAT0Y, -1),  # Example: Move Up
                9: lambda: self.press_abs(e.ABS_HAT0Y, 1),   # Example: Move Down
                10: lambda: self.press_button(e.BTN_SELECT), # Press SELECT
                12: lambda: self.press_button(e.BTN_START)   # Press START (Needed for traversing planets on it's own)
            }
        
            # Execute the corresponding method for the predicted action
            action_method = action_methods.get(action, lambda: None)
            action_method()
        thread = threading.Thread(target=action_to_controller, args=(action,))
        thread.daemon = True
        thread.start()

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(image)  # Convert array to PIL Image
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

def start_screen_capture():
    def get_star_fox():
        title = "Star Fox (USA) (Rev 2)"
        try:
            while True:
                # Use xwininfo to get geometry of the window
                proc = subprocess.Popen(['xwininfo', '-name', title], stdout=subprocess.PIPE)
                output, _ = proc.communicate()
                output = output.decode()

                # Extract geometry
                x = int(re.search('Absolute upper-left X: +(\d+)', output).group(1))
                y = int(re.search('Absolute upper-left Y: +(\d+)', output).group(1))
                width = int(re.search('Width: +(\d+)', output).group(1))
                height = int(re.search('Height: +(\d+)', output).group(1))
        
                # Capture the window directly using 'import' command from ImageMagick
                cmd = f"import -window '{title}' png:-"
                img_data = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)  # Redirect stderr to /dev/null
    
                # Convert the captured data to an image and preprocess
                image = Image.open(io.BytesIO(img_data))
                img = np.array(image)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_tensor = preprocess_image(img)  # Preprocess the image
            
                # Use the model to predict the next action
                with torch.no_grad():
                    print("Predicting action...")
                    predictions = model(img_tensor.to('cuda:0'))
                    action = predictions.argmax().item()  # Choose the action with the highest score                
                    # Map the predicted action to a controller command
                    vc.execute_action(action)
                print("Action predicted.")
        except Exception as e:
            print(f"Failed to capture game window directly: {e}")
        finally:
            cv2.destroyAllWindows()
    
    thread = threading.Thread(target=get_star_fox)
    thread.daemon = True
    thread.start()
        
# GUI Application
class App:
    def __init__(self, root, controller):
        self.controller = controller
        root.title("Virtual Controller GUI")
        tk.Button(root, text="Start Configuration", command=self.controller.start_auto_conf).pack()
        tk.Button(root, text="Start Screen Capture (Init FoxAI)", command=start_screen_capture).pack()
        
if __name__ == "__main__":
    model = FoxAI()
    model.to('cuda:0')
    model.eval()  # Set the model to evaluation mode
    # Configures the virtual controller for later use by FoxAI (The SNES agent that plays StarFox, studies are on 3d interpretation by AI.)
    vc = VirtualController()
    root = tk.Tk()
    app = App(root, vc)
    root.mainloop()

    # TODO: Implement training logic, make discriminator model, train discriminator model on good and bad states, use discriminator model to evaluate action model (FoxAI)
