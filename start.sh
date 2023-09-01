#!/bin/sh

conda create -n function_translator
conda activate function_translator
nohup flask --app octopus_function_translator run -h 0.0.0.0 -p 5001 2>&1 &
conda deactivate

conda create -n function_orbit_camera
conda activate function_orbit_camera
nohup flask --app octopus_plugin_blender_orbit_camera run -h 0.0.0.0 -p 5002 2>&1 &
conda deactivate

python3 -m venv .venv
. .venv/bin/activate
nohup flask --app octopus_visual_questions_answering run -h 0.0.0.0 -p 5003 2>&1 &

conda create -n function_sensitive_information
conda activate function_sensitive_information
nohup flask --app octopus_DataPrivacyComplianceCheck run -h 0.0.0.0 -p 5004 2>&1 &
conda deactivate

python3 -m venv .venv
. .venv/bin/activate
nohup flask --app sdxl10_octopusapi run -h 0.0.0.0 -p 5005 2>&1 &

python3 -m venv .venv
. .venv/bin/activate
nohup flask --app information_retrieval run -h 0.0.0.0 -p 5006 2>&1 &
