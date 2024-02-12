# -*- coding: utf-8 -*-
"""SDXL10_octopusAPI_orbit_camera_linux.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P6fWDskpJfEc7hNM9IgUPz8dL_wVB44V
"""

import os
import argparse

### BEGIN USER EDITABLE SECTION ###
dependencies = [
    "pip install -q flask==3.0.0",
    "pip install -q pyngrok==7.0.0",
    "pip install -q joblib==1.3.2",
    "pip install -q uuid==1.30",
    "pip install nc_py_api==0.8.0",
    "apt-get update --fix-missing && apt-get install -y --no-install-recommends libgl1-mesa-glx libsm6 libxkbcommon-x11-0 libxi6 libxxf86vm1"
]

for command in dependencies:
    os.system(command)

from nc_py_api import Nextcloud

nc_user = os.getenv('NC_USERNAME')
nc_password = os.getenv('NC_PASSWORD')

bpy_package = "bpy-3.6.3rc0-cp310-cp310-manylinux_2_35_x86_64.whl"

nc = Nextcloud(nextcloud_url="https://nx47886.your-storageshare.de", nc_auth_user=nc_user, nc_auth_pass=nc_password)
nc.files.download2stream("auditron_augmentation/bpy-3.6.3rc0-cp310-cp310-manylinux_2_35_x86_64.whl", bpy_package)
os.system(f"pip install {bpy_package}")

config_str = '''
 {
    "device_map": {
     "cuda:1": "10GiB",
     "cpu": "30GiB"
     },
     "models": [
         {
             "key": "10133_integration.blend",
             "name": "blender_model",
             "access_token": "orbit_camera/"
         }
     ],
     "functions": [
         {
             "name": "orbit-camera",
             "description": "This function renders a .png image of a 3D model of AFK machine. You can control from what point in space the image is taken. The camera orbits around the machine, and its position is controlled by position argument. This function is useful to get an overview of the AFK machine from a given angle or a view of a particular section of the machine.",
             "parameters": {
                 "type": "object",
                 "properties": {
                     "position":{
                         "type": "int",
                         "description": "This argument values are from 1-100 and represent 360 degrees rotation around the machine. A value of 1 means that the machine is viewed from the front, 25 from the right, 50 from the back, and 75 from the left."
                     },
                     "tilt":{
                         "type": "string",
                         "enum": ["up", "center", "down"],
                         "description": "This argument controls the camera rotation in x axis, when the camera is tilted down machine is shown from top etc. That allows seeing the AFK machine from different perspectives."
                     },
                     "zoom":{
                         "type": "string",
                         "enum": ["far", "middle", "close"],
                         "description": "This argument controls the camera zoom, it can be used to look at the machine closer or farther away to get more context. Can be used for getting closed-up shots of particular sections of the AFK machine."
                     },
                     "resolution":{
                         "type": "string",
                         "enum": ["full-hd", "4k"],
                         "description": "This argument allows to select the resolution desired by the user, 4k resolution helps to see finer detail on the rendered image, in case some fine details are too small to be seen correctly on a standard full HD image."
                     }
                 },
                 "required": ["position"]
             },
             "input_type": "json",
             "return_type": "image/png"
         },
         {
             "name": "find-part",
             "description": "This function renders a .png image, showing the selected part in full view, of a 3D model of AFK machine. The part gets highlighted with a magenta color. The camera is floating on one side of the machine, and it is moved and zoomed in to the selected part. This function is helpful for getting to know how parts of the AFK machines look and how are mounted or placed on the machine.",
             "parameters": {
                 "type": "object",
                 "properties": {
                     "part":{
                         "type": "string",
                         "description": "This argument is the part name to be found and shown on render, it must exactly match the part name in 3D model. Otherwise the function will not return the rendered image"
                     },
                     "clip":{
                         "type": "boolean",
                         "description": "This argument makes all objects between the camera and the given object not visible on the rendered image if set to True. It allows showing the requested part in case it is not visible on the render because it is obstructed by other parts of the machine."
                     },
                     "resolution":{
                         "type": "string",
                         "enum": ["full-hd", "4k"],
                         "description": "This argument allows to select the resolution desired by the user, 4k resolution helps to see finer detail on the rendered image, in case some fine details are too small to be seen correctly on a standard full HD image."
                     }
                 },
                 "required": ["part"]
             },
             "input_type": "json",
             "return_type": "image/png"
         }

     ]
 }
'''

import bpy  #Starts Blender instance
import math
import mathutils

class BlenderModelManager:
    def __init__(self, key, name, token, gpu_index = 0):
        extension = key.split(".")[-1]
        file_name = f"{name}.{extension}"
        nc.files.download2stream(token+key, file_name)

        bpy.ops.wm.open_mainfile(filepath=file_name)
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.refresh_devices()
        bpy.context.preferences.addons['cycles'].preferences.devices[gpu_index].use = True
        print(f"Using GPU:{gpu_index} for Blender rendering")
        bpy.context.scene.cycles.device = 'GPU'

        self.orbit_camera_object = bpy.data.objects.get("Orbit_camera")
        self.orbit_curve_object = bpy.data.objects.get("Orbit_curve")
        self.part_camera_object = bpy.data.objects.get("Part_camera")

        print("Blender model loaded")
    
    def orbit_inference(self, output_filename, position, tilt = "center", zoom ="middle", resolution = "full-hd"):
        self._orbit(position, tilt, zoom)
        self._generate_image(self.orbit_camera_object, resolution, output_filename)
    
    def find_part_inference(self, output_filename, part_name, resolution = "full-hd", clip=False):
        self._find_part(self.part_camera_object, part_name, clip)
        self._change_object_material(part_name, "highlighter")
        self._generate_image(self.part_camera_object, resolution, output_filename)
        self._restore_object_material(part_name)
        
    def _orbit(self, position:int, tilt, zoom):
        self._change_tilt(tilt)
        self._change_zoom(zoom)
        self._change_position(position)
    
    def _find_part(self,camera, part_name, clip):
        distance = self._copy_object_center_to_camera(part_name, camera)
        if clip is True:
            self._set_camera_clip(camera, distance * 0.9)
        else:
            self._set_camera_clip(camera, 0.1)

    def _generate_image(self, camera, resolution="full-hd", filename="output"):
        assert camera and camera.type == "CAMERA", "given object for renderign an image is not a camera object"
        x_res_dict = {
        "full-hd" : 1920,
        "4k" : 3840
        }
        y_res_dict = {
        "full-hd" : 1080,
        "4k" : 2160
        }
        bpy.context.scene.render.resolution_x = x_res_dict[resolution]
        bpy.context.scene.render.resolution_y = y_res_dict[resolution]
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = f"//{filename}.png"  # Relative to blend file

        bpy.context.scene.camera = camera
        bpy.ops.render.render(write_still=True)

    def _change_tilt(self, tilt="center"):
        assert self.orbit_curve_object and self.orbit_curve_object.type == "CURVE", "given object for tilt manipulation is not a curve object"
        curve = self.orbit_curve_object.data
        
        tilt_dict = {
            "down" : 25.0,
            "center" : 5.0,
            "up"  : -5.0
        }

        # Access the control points of the curve
        points = curve.splines[0].bezier_points
        
        # Modify the tilt of control points
        for point in points:
            point.tilt = math.radians(tilt_dict[tilt])

    def _change_zoom(self, zoom="middle"):
        assert self.orbit_camera_object and self.orbit_camera_object.type == "CAMERA", "given object for zoom manipulation is not a camera object"
        zoom_dict = {
            "far" : 9,
            "middle" : 22,
            "close" : 28
        }
        self.orbit_camera_object.data.lens = zoom_dict[zoom]

    def _change_position(self, position):
        follow_path_constraint = self.orbit_camera_object.constraints.get("Follow Path")
        assert follow_path_constraint and follow_path_constraint.type == 'FOLLOW_PATH', "follow path constraint not found for given object for position manipulation"
        follow_path_constraint.offset = position

    def _find_geometric_center(self, object_name):
        # Find the object by name
        target_object = bpy.data.objects.get(object_name)

        assert target_object is not None, "Object given for geometry center does not exist, check the part name given"
        # Get the object's bounding box
        verts = target_object.data.vertices
        assert verts is not None and len(verts) > 0, "Object given for geometry center does not have verticles, but it exists it might be an empty type object, try another name" 
        bbox = [target_object.matrix_world @ v.co for v in verts]

        # Calculate the geometric center
        geometric_center = sum(bbox, mathutils.Vector()) / len(bbox)

        return geometric_center

    def _set_camera_clip(self, camera, near_clip):
        assert camera and camera.type == "CAMERA", "given object for clip manipulation is not a camera object"
        # Set the near and far clip values
        camera.data.clip_start = near_clip

    def _copy_object_center_to_camera(self, object_name, camera):
        # Find the object by name
        target_object = bpy.data.objects.get(object_name)

        assert target_object is not None, "Part name given does not exist in Blender file, check if correct name was given"

        # Get the X coordinate of the object's center
        coordinate = self._find_geometric_center(object_name)
        x_coordinate = coordinate.x

        # Set the camera's X coordinate of location
        camera.location.x = x_coordinate

        # Calculate the camera's distance to the object's center
        distance_to_object = (camera.location - coordinate).length

        #Calculate focal length to keep the object in the frame
        sensor_width = camera.data.sensor_width/1000
        sensor_height = camera.data.sensor_height/1000
        object_width = target_object.dimensions.x * 3
        object_height = target_object.dimensions.z * 3
        desired_focal_length_width = ((distance_to_object*sensor_width) / object_width) * 1000
        desired_focal_length_height = ((distance_to_object*sensor_height) / object_height) * 1000

        # Use the smaller of the two focal lengths to ensure the entire object fits in the frame
        desired_focal_length = min(desired_focal_length_width, desired_focal_length_height)

        # Set the camera's focal length
        camera.data.lens = desired_focal_length

        #tilt the camera to look at the object
        direction_to_target = coordinate - camera.location
        tilt_angle = math.atan2(direction_to_target.z, direction_to_target.y)
        camera.rotation_euler.x= tilt_angle+math.radians(90)

        # Update the scene to refresh the camera view
        bpy.context.view_layer.update()

        return distance_to_object
    
    def _change_object_material(self, object_name, new_material_name):
        # Find the object by name
        selected_object = bpy.data.objects.get(object_name)
        assert selected_object is not None, "Object given for highlighting doesn't exist"
        # Store the initial material(s) of the object
        self.initial_materials = selected_object.data.materials[:]
        # Get an existing material by its name
        existing_material = bpy.data.materials.get(new_material_name)
        if existing_material:
            # Assign the existing material to the object
            selected_object.data.materials[0] = existing_material  # Replace the first material slot
            selected_object.data.update()
    def _restore_object_material(self, object_name):
        # Restore the initial materials (assuming the number of material slots matches)
        selected_object = bpy.data.objects.get(object_name)
        for i, material in enumerate(self.initial_materials):
            selected_object.data.materials[i] = material

        # Update the object to reflect the material changes
        selected_object.data.update()
            
try:
    import google.colab
    # Change to /content directory
    os.chdir('/content')
    print("Changed directory to /content")
except ImportError:
    print("Not running in Colab")

### END USER EDITABLE SECTION ###

import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify
import uuid
import time

config = json.loads(config_str)
app = Flask(__name__)

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.device = self.select_device()
        self.error_image = None

    def select_device(self):
        device_map = self.config.get('device_map', {})
        available_devices = list(device_map.keys())
        return available_devices[0] if available_devices else "cpu"

    def setup(self):
        self.models.clear()

        # Loading models from configuration
        for model_info in self.config["models"]:
            model_key = model_info["key"]
            model_name = model_info["name"]
            model_access_token = model_info["access_token"]
            try:
                ### BEGIN USER EDITABLE SECTION ###
                gpu_index = self._get_gpu()
                model = BlenderModelManager(model_key,model_name, model_access_token, gpu_index)
                self.models[model_name] = model
                ### END USER EDITABLE SECTION ###
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")

    def infer_orbit(self, parameters):
        try:
            ### BEGIN USER EDITABLE SECTION ###
            output_filename = str(uuid.uuid4())
            blender_model = self.models["blender_model"]

            print("orbit-inference")
            blender_model.orbit_inference(output_filename, position=parameters.get('position'), tilt=parameters.get('tilt', "center"), zoom=parameters.get('zoom', "middle"), resolution=parameters.get('resolution', "full-hd"))

            while os.path.exists(output_filename+".png") == False:
                time.sleep(5)

            return output_filename+".png", "Here is the image for you!"
            ### END USER EDITABLE SECTION ###
        except Exception as e:
            print(f"Error during inference: {e}")
            return self.error_image, e

    def infer_part(self, parameters):
        try:
            ### BEGIN USER EDITABLE SECTION ###
            output_filename = str(uuid.uuid4())
            blender_model = self.models["blender_model"]

            print("part-inference")
            blender_model.find_part_inference(output_filename, part_name=parameters.get('part'), resolution=parameters.get('resolution', "full-hd"), clip=parameters.get('clip', False))

            while os.path.exists(output_filename+".png") == False:
                time.sleep(5)

            return output_filename+".png", "Here is the image for you!"
        ### END USER EDITABLE SECTION ###
        except Exception as e:
            print(f"Error during inference: {e}")
            return self.error_image, e
    
    def _get_gpu(self):
        min_cuda_key = None
        min_cuda_value = float('inf')  # Initialize with positive infinity
        
        required_memory_gb = 6.0

        for key, value in config["device_map"].items():
            if key.startswith("cuda:") and "GiB" in value:
                memory_value = float(value.replace("GiB", ""))
                if memory_value > required_memory_gb and memory_value < min_cuda_value:
                    min_cuda_key = key
                    min_cuda_value = memory_value
        if min_cuda_key:
            print(f"CUDA with the smallest value above {required_memory_gb}GB: {min_cuda_key}")
            print(f"Memory value: {min_cuda_value}GiB")
            return int(min_cuda_key.split(":")[-1])
        else:
            raise Exception("No CUDA found with a value above 4GB.")

model_manager = ModelManager(config)

@app.route('/v1/setup', methods=['POST'])
def setup():
    model_manager.setup()
    return jsonify({"status": "models loaded successfully", "setup": "Performed"}), 201

@app.route('/v1/orbit-camera', methods=['POST'])
def generic_route():
    function_config = config["functions"][0]

    if not function_config:
        return jsonify({"error": "Invalid endpoint"}), 404

    if function_config["input_type"] != "json":
        return jsonify({"error": f"Unsupported input type {function_config['input_type']}"}), 400

    data = request.json
    parameters = {k: data[k] for k in function_config["parameters"]["properties"].keys() if k in data}

    result, header = model_manager.infer_orbit(parameters)
    if result:
        file = open(result, mode='rb')
        fcontent = file.read()
        file.close()
        return app.response_class(fcontent, content_type=function_config["return_type"]), 201
    else:
        return jsonify({"error": "Error during inference"}), 500

@app.route('/v1/find-part', methods=['POST'])
def generic_route_1():
    function_config = config["functions"][1]

    if not function_config:
        return jsonify({"error": "Invalid endpoint"}), 404

    if function_config["input_type"] != "json":
        return jsonify({"error": f"Unsupported input type {function_config['input_type']}"}), 400

    data = request.json
    parameters = {k: data[k] for k in function_config["parameters"]["properties"].keys() if k in data}

    result, header = model_manager.infer_part(parameters)
    if result:
        file = open(result, mode='rb')
        fcontent = file.read()
        file.close()
        return app.response_class(fcontent, content_type=function_config["return_type"]), 201
    else:
        return jsonify({"error": "Error during inference"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Generic exception handler
    return jsonify(error=str(e)), 500

from pyngrok import ngrok
import time

parser = argparse.ArgumentParser(description="Service using blender to generate images for chat gpt")
parser.add_argument('--host', type=str, default="127.0.0.1",help='set the host for service')
parser.add_argument('--port', type=int, default = "5000", help="set the port for the service")
args = parser.parse_args()

# Start the Flask server in single threaded mode
app.run(host = args.host, port = args.port, threaded=False)

# Set up Ngrok to create a tunnel to the Flask server
public_url = "http://127.0.0.1:5002"
public_url = ngrok.connect(5000).public_url
#public_url = "http://127.0.0.1:5002"

function_names = [func['name'] for func in config["functions"]]

print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{5000}/\"")

# Loop over function_names and print them
for function_name in function_names:
   time.sleep(5)
   print(f'Endpoint here: {public_url}/{function_name}')
