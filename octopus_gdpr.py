import os

dependencies = [
    "pip install -q Flask==3.1.0",
    "pip install -q torch==2.6.0",
    "pip install -q transformers==4.48.2"
]

for command in dependencies:
    os.system(command)

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import torch
import subprocess
from flask import Flask, jsonify, request

config_str = '''{
    "device_map": {
        "cuda:0": "10GiB",
        "cpu": "5GiB"
    },
    "required_python_version": "cp312",
    "models": {
        "key": "metricspace/GDPR_Input_Detection_and_Anonymization_0.5B",
        "name": "base_model"
    },
    "tokenizer": {
        "key": "metricspace/GDPR_Input_Detection_and_Anonymization_0.5B",
        "name": "tokenizer"
    },
    "functions": [
        {
            "name": "anonymization",
            "display_name": "Anonymization",
            "description": "The function Anonymization is designed to extract and anonymize sensitive entities from a provided text. This function is highly effective for protecting personal or sensitive information by replacing identifiable data with anonymized alternatives, ensuring privacy and confidentiality. It should be triggered when there’s a need to anonymize specific data such as names, addresses, or other personal identifiers within various texts, including legal documents, research papers, and customer feedback. Examples of when to use this function include: (1) anonymizing personal details in legal contracts, (2) removing sensitive information from research data before publication, and (3) anonymizing customer feedback to prevent data breaches. This function should not be triggered for scenarios where anonymization is not required, such as tasks that involve (1) generic text processing, (2) general information extraction, or (3) summarizing non-sensitive content for general insights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value1": {
                        "type": "string",
                        "description": "Text for anonymization"
                    }
                },
                "required": ["value1"]
            },
            "input_type": "application/json",
            "return_type": "application/json"
        },
        {
            "name": "llm_router",
            "display_name": "LLM router",
            "description": "The function LLM router returns info about LLM that was selected for answearing user question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text"
                    }
                },
                "required": ["text"]
            },
            "input_type": "application/json",
            "return_type": "application/json"
        },
        {
            "name": "sensitive_information",
            "display_name": "Sensitive information filter",
            "description": "Predicts if the user input is sensitive and shows sensitive info",
            "parameters": {
                "type": "object",
                "properties": {
                    "value1": {
                        "type": "string",
                        "description": "prompt - check for sensitive information"
                    }
                },
                "required": ["value1"]
            },
            "input_type": "application/json",
            "return_type": "application/json"
        }
    ]
}'''

config = json.loads(config_str)
app = Flask(__name__)

formats = {
    "sensitivity": """<|im_start|>system\nSensitivity<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n""",
    "complexity": """<|im_start|>system\nComplexity<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n""",
    "entity_detection": """<|im_start|>system\nEntity Detection<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n""",
    "entity_swapping": """<|im_start|>system\nEntity Swapping<|im_end|>\n<|im_start|>user\nentities:\n{entities}\ntext:\n{text}<|im_end|>\n<|im_start|>assistant\n"""
}

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.device = self.select_device()

    def command_result_as_int(self, command):
        return int(subprocess.check_output(command, shell=True).decode('utf-8').strip())

    def select_device_with_larger_free_memory(self, available_devices):
        device = None
        memory = 0

        for available_device in available_devices:
            id = available_device.split(":")
            id = id[-1]
            try:
                free_memory = self.command_result_as_int(f"nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader --id={id}")
                if free_memory > memory:
                    memory = free_memory
                    device = available_device
            except:
                print(f"problem with executing nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader --id={id}")

        return device if device else "cpu"

    def select_device(self):
        if not torch.cuda.is_available():
            return "cpu"

        device_map = self.config.get('device_map', {})
        available_devices = list(device_map.keys())
        print(available_devices)
        return self.select_device_with_larger_free_memory(available_devices)

    def setup(self):
        self.models.clear()
        model_info = self.config["models"]
        model_key = model_info["key"]
        model_name = model_info["name"]

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_key)
            model = AutoModelForCausalLM.from_pretrained(model_key, torch_dtype=torch.float16).to(self.device)

            tokenizer.pad_token = "<|im_start|>"
            tokenizer.eos_token = "<|im_end|>"
            tokenizer.padding_side = "left"
            model.generation_config.pad_token_id = tokenizer.pad_token_id

            self.models['base_model'] = model
            self.models['tokenizer'] = tokenizer
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")

    def model_inference(self, text, mode="anonymization", max_new_tokens=2028, config=None, entity_mapping=None, return_entities=False, reverse_mapping=False):
        if mode not in formats and mode != "anonymization":
            raise ValueError("Invalid mode. Choose from 'sensitivity', 'complexity', 'entity_detection', 'anonymization'.")

        # Configuration for anonymization
        # The `config` dictionary specifies the anonymization behavior for each type of entity detected.
        # Each key in `config` represents an entity type (e.g., "LOC" for location, "PERSON" for personal names),
        # and the value assigned to that key determines how entities of that type should be anonymized:
        #
        # - "RANDOM": Replaces the entity with a randomly selected placeholder.
        # - "GENERAL LOW", "GENERAL MEDIUM", "GENERAL HIGH": Replaces the entity with a generalized label,
        #   with the intensity level (LOW, MEDIUM, HIGH) controlling the specificity. For example, 
        #   "GENERAL LOW" might use a more specific label ("Local Park") while "GENERAL HIGH" would use
        #   a broader label ("Recreational Area").
        #
        # This allows fine-grained control over anonymization, ensuring that different types of sensitive 
        # information can be replaced in ways that are appropriate for the context. For example:
        #   - "LOC": "RANDOM" replaces any detected location with a random placeholder.
        #   - "DATETIME": "GENERAL LOW" uses a lower-intensity generalization for dates and times.
        #
        # This flexibility enables custom anonymization policies to suit different privacy or obfuscation needs.

        if config is None:
            config = {
                "LOC": "RANDOM",
                "PERSON": "RANDOM",
                "DEM": "RANDOM",
                "CODE": "RANDOM",
                "ORG": "GENERAL MEDIUM",
                "DATETIME": "GENERAL LOW",
                "QUANTITY": "RANDOM",
                "MISC": "RANDOM",
            }

        model = self.models['base_model']
        tokenizer = self.models['tokenizer']

        # Anonymization Mode
        if mode == "anonymization":
            # Step 1: Entity detection
            detection_prompt = formats["entity_detection"].format(text=text)
            detection_inputs = tokenizer(detection_prompt, return_tensors="pt").to(self.device)
            detection_output = model.generate(
                **detection_inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                eos_token_id=151645
            )
            detection_text = tokenizer.decode(detection_output[0], skip_special_tokens=True)
            detected_entities = self.postprocess_entity_recognition(detection_text)

            # Step 2: Select entities based on config
            selected_entities = self.select_entities_based_on_json(detected_entities, config)
            entities_str = "\n".join([f"{entity} : {label}" for entity, label in selected_entities])
            # Step 3: Entity swapping for anonymization
            swapping_prompt = formats["entity_swapping"].format(entities=entities_str, text=text)
            swapping_inputs = tokenizer(swapping_prompt, return_tensors="pt").to(self.device)
            swapping_output = model.generate(
                **swapping_inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                eos_token_id=151645
            )

            anonymized_text = tokenizer.decode(swapping_output[0], skip_special_tokens=True)
            anonymized_text = anonymized_text.split("assistant\n", 1)[-1].strip()  # Extract only the assistant's response

            if return_entities:
                return anonymized_text, entities_str

            return anonymized_text

        # Entity Restoration Mode using entity_swapping
        elif mode == "entity_swapping" and entity_mapping:
            # Reverse the entity mapping
            if reverse_mapping:
                reversed_mapping = []
                for line in entity_mapping.splitlines():
                    if ':' in line:  # Ensure the line contains a colon
                        left, right = map(str.strip, line.split(":", 1))  # Split and strip spaces
                        reversed_mapping.append(f"{right} : {left}")  # Reverse and format
                entity_mapping = "\n".join(reversed_mapping)

            # Create the swapping prompt with the aggregated reversed mappings
            swapping_prompt = formats["entity_swapping"].format(entities=entity_mapping, text=text)
            swapping_inputs = tokenizer(swapping_prompt, return_tensors="pt").to(self.device)
            swapping_output = model.generate(
                **swapping_inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                eos_token_id=151645
            )

            # Decode and extract the restored text
            output_text = tokenizer.decode(swapping_output[0], skip_special_tokens=True)
            output_text = output_text.split("assistant\n", 1)[-1].strip()  # Extract only the assistant's response

            return output_text

        # Other modes (sensitivity, complexity, entity_detection)
        else:
            prompt = formats[mode].format(text=text)
            model_inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            generation_output = model.generate(
                **model_inputs,
                max_new_tokens=5,
                use_cache=True,
                eos_token_id=151645
            )
            full_output = tokenizer.decode(generation_output[0], skip_special_tokens=True)

            if mode in ["sensitivity", "complexity"]:
                assistant_text = full_output.split("assistant\n", 1)[-1].strip()
                return assistant_text
            elif mode == "entity_detection":
                return self.postprocess_entity_recognition(full_output)

    # Function to parse entity detection output
    def postprocess_entity_recognition(self, detection_output: str) -> dict:
        output_json = {}
        entity_pattern = re.compile(
            r'(?P<entity>[\w\s]+)--(?P<type>[\w]+)--(?P<random>[\w\s]+)--(?P<generalizations>.+)'
        )
        generalization_pattern = re.compile(r'([\w\s]+)::([\w\s]+)')

        lines = detection_output.strip().split("\n")
        for line in lines:
            match = entity_pattern.search(line)
            if match:
                entity_name = match.group("entity").strip()
                entity_type = match.group("type").strip()
                random_replacement = match.group("random").strip()

                generalizations = []
                for gen_match in generalization_pattern.findall(match.group("generalizations")):
                    first, second = gen_match

                    # Check if the first part is a digit (score) and swap if needed
                    if first.isdigit() and not second.isdigit():
                        score = first
                        label = second
                        generalizations.append([label.strip(), score.strip()])

                    elif not first.isdigit() and second.isdigit():
                        label = first
                        score = second
                        generalizations.append([label.strip(), score.strip()])

                output_json[entity_name] = {
                    "TYPE": entity_type,
                    "RANDOM": random_replacement,
                    "GENERAL": generalizations
                }
        return output_json

    # Function to select entities based on config
    def select_entities_based_on_json(self, prediction_json, entity_json):
        entities = []
        for key, value in prediction_json.items():
            entity_type = value["TYPE"]
            if entity_type.upper() in entity_json:
                anonymization_type = entity_json[entity_type]
                if anonymization_type == "RANDOM":
                    entities.append([key, value["RANDOM"]])
                elif "GENERAL" in anonymization_type:
                    intensity = anonymization_type.split(" ")[1]
                    if intensity == "LOW" and value["GENERAL"]:
                        entities.append([key, value["GENERAL"][0][0]])
                    elif intensity == "MEDIUM":
                        for gen in value["GENERAL"]:
                            if int(gen[1]) >= 4:
                                entities.append([key, gen[0]])
                                break
                    elif intensity == "HIGH":
                        if value["GENERAL"]:
                            entities.append([key, value["GENERAL"][0][0]])
        return entities

model_manager = ModelManager(config)

@app.route('/v1/setup', methods=['POST'])
def setup():
    model_manager.setup()
    return jsonify({"setup": "Performed"}), 201

@app.route('/v1/anonymization', methods=['POST'])
def anonymization():
    data = request.json
    text = data.get("value1", "")

    result = model_manager.model_inference(text, "anonymization")

    response = {
        "response": result
    }

    return jsonify(response), 201

@app.route('/v1/llm_router', methods=['POST'])
def llm_router():
    data = request.json
    text = data.get("text", "")

    result = model_manager.model_inference(text, "complexity")

    response = {
        "response": result
    }

    return jsonify(response), 201

@app.route('/v1/sensitive_information', methods=['POST'])
def sensitive_information():
    data = request.json
    text = data.get("value1", "")

    result = model_manager.model_inference(text, "sensitivity")
    result_int = int(result)

    if result_int >= 2:
        response = {
            "response": result,
            "is_sensitive": True,
            "sensitive_part": None
        }
    else:
        response = {
            "response": result,
            "is_sensitive": False,
            "sensitive_part": None
        }

    return jsonify(response), 201

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
