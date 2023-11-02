#!/bin/sh

sudo ufw allow 5001/tcp
python3 -m venv .venv
. .venv/bin/activate
nohup flask --app octopus_function_translator run -h 0.0.0.0 -p 5001 2>&1 &

sudo ufw allow 5002/tcp
python3 -m venv .venv
. .venv/bin/activate
nohup python3 octopus_plugin_blender_orbit_camera.py --host=0.0.0.0 --port=5002 2>&1 &

sudo ufw allow 5003/tcp
python3 -m venv .venv
. .venv/bin/activate
nohup flask --app octopus_visual_questions_answering run -h 0.0.0.0 -p 5003 2>&1 &

sudo ufw allow 5004/tcp
python3 -m venv .venv
. .venv/bin/activate
nohup flask --app octopus_DataPrivacyComplianceCheck run -h 0.0.0.0 -p 5004 2>&1 &

sudo ufw allow 5005/tcp
python3 -m venv .venv
. .venv/bin/activate
nohup flask --app octopus_text2img_SDXL run -h 0.0.0.0 -p 5005 2>&1 &

python3 -m venv .venv
. .venv/bin/activate
nohup flask --app information_retrieval run -h 0.0.0.0 -p 5006 2>&1 &


# octopus_function_translator.py
# cuda:0 - 7.0 GiB

# octopus_DataPrivacyComplianceCheck.py
# cuda:0 - 6.0 GiB

# octopus_text2img_SDXL.py
# cuda:1 - 11.5 GiB

# octopus_plugin_blender_orbit_camera.py
# cuda:3 - 6.0 GiB
