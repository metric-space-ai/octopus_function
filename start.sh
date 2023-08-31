#!/bin/sh

nohup flask --app octopus_function_translator run -h 0.0.0.0 -p 5001 2>&1 &
nohup flask --app octopus_plugin_blender_orbit_camera run -h 0.0.0.0 -p 5002 2>&1 &
nohup flask --app octopus_visual_questions_answering run -h 0.0.0.0 -p 5003 2>&1 &
nohup flask --app octopus_DataPrivacyComplianceCheck run -h 0.0.0.0 -p 5004 2>&1 &
nohup flask --app sdxl10_octopusapi run -h 0.0.0.0 -p 5005 2>&1 &
