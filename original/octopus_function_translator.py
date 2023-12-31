
#TODO: list_of_dependencies(...)
import json, threading, uuid
from datetime import datetime, timedelta
from flask import Flask, jsonify, request

import requests
import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import fasttext
import nltk
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

### Configuration section
config = '''{
    "device_map": {
       "max_memory": {"0": "10GiB", "1": "10GiB", "cpu": "30GiB"}
    },
    "model_setup": {
        "file": "lid218e.bin",
        "url": "https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin"
        TODO: add the model
    },
    "functions": [
        {
            "name": "function_translator",
            "url_part": "function-translator",
            "description": "Translator function",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_language": {
                        "type": "string",
                        "description": "Source language of the text"
                    },
                    "target_language": { "type": "string", "description": "Target language for translation" },
                    "text": { "type": "string", "description": "Translated text" }
                },
                "required": ["source_language", "target_language", "text"]
            }
        }
    ]}'''
config = json.loads(config)

### Utils section
### TODO: why this, it'S bloating the template
def get_estimated_response_at(seconds: int) -> str:
    estimated_response_at = str(datetime.now() + timedelta(seconds=seconds))
    estimated_response_at = estimated_response_at + "Z"
    estimated_response_at = estimated_response_at.replace(" ", "T")

    return estimated_response_at

### AI function section
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #TODO: it won't not work like this, therefor we have the device map defintion
file = config["model_setup"]["file"]
url = config["model_setup"]["url"]
flores_codes={'Acehnese (Arabic script)': 'ace_Arab', 'Acehnese (Latin script)': 'ace_Latn', 'Mesopotamian Arabic': 'acm_Arab', 'Ta’izzi-Adeni Arabic': 'acq_Arab', 'Tunisian Arabic': 'aeb_Arab', 'Afrikaans': 'afr_Latn', 'South Levantine Arabic': 'ajp_Arab', 'Akan': 'aka_Latn', 'Amharic': 'amh_Ethi', 'North Levantine Arabic': 'apc_Arab', 'Modern Standard Arabic': 'arb_Arab', 'Modern Standard Arabic (Romanized)': 'arb_Latn', 'Najdi Arabic': 'ars_Arab', 'Moroccan Arabic': 'ary_Arab', 'Egyptian Arabic': 'arz_Arab', 'Assamese': 'asm_Beng', 'Asturian': 'ast_Latn', 'Awadhi': 'awa_Deva', 'Central Aymara': 'ayr_Latn', 'South Azerbaijani': 'azb_Arab', 'North Azerbaijani': 'azj_Latn', 'Bashkir': 'bak_Cyrl', 'Bambara': 'bam_Latn', 'Balinese': 'ban_Latn', 'Belarusian': 'bel_Cyrl', 'Bemba': 'bem_Latn', 'Bengali': 'ben_Beng', 'Bhojpuri': 'bho_Deva', 'Banjar (Arabic script)': 'bjn_Arab', 'Banjar (Latin script)': 'bjn_Latn', 'Standard Tibetan': 'bod_Tibt', 'Bosnian': 'bos_Latn', 'Buginese': 'bug_Latn', 'Bulgarian': 'bul_Cyrl', 'Catalan': 'cat_Latn', 'Cebuano': 'ceb_Latn', 'Czech': 'ces_Latn', 'Chokwe': 'cjk_Latn', 'Central Kurdish': 'ckb_Arab', 'Crimean Tatar': 'crh_Latn', 'Welsh': 'cym_Latn', 'Danish': 'dan_Latn', 'German': 'deu_Latn', 'Southwestern Dinka': 'dik_Latn', 'Dyula': 'dyu_Latn', 'Dzongkha': 'dzo_Tibt', 'Greek': 'ell_Grek', 'English': 'eng_Latn', 'Esperanto': 'epo_Latn', 'Estonian': 'est_Latn', 'Basque': 'eus_Latn', 'Ewe': 'ewe_Latn', 'Faroese': 'fao_Latn', 'Fijian': 'fij_Latn', 'Finnish': 'fin_Latn', 'Fon': 'fon_Latn', 'French': 'fra_Latn', 'Friulian': 'fur_Latn', 'Nigerian Fulfulde': 'fuv_Latn', 'Scottish Gaelic': 'gla_Latn', 'Irish': 'gle_Latn', 'Galician': 'glg_Latn', 'Guarani': 'grn_Latn', 'Gujarati': 'guj_Gujr', 'Haitian Creole': 'hat_Latn', 'Hausa': 'hau_Latn', 'Hebrew': 'heb_Hebr', 'Hindi': 'hin_Deva', 'Chhattisgarhi': 'hne_Deva', 'Croatian': 'hrv_Latn', 'Hungarian': 'hun_Latn', 'Armenian': 'hye_Armn', 'Igbo': 'ibo_Latn', 'Ilocano': 'ilo_Latn', 'Indonesian': 'ind_Latn', 'Icelandic': 'isl_Latn', 'Italian': 'ita_Latn', 'Javanese': 'jav_Latn', 'Japanese': 'jpn_Jpan', 'Kabyle': 'kab_Latn', 'Jingpho': 'kac_Latn', 'Kamba': 'kam_Latn', 'Kannada': 'kan_Knda', 'Kashmiri (Arabic script)': 'kas_Arab', 'Kashmiri (Devanagari script)': 'kas_Deva', 'Georgian': 'kat_Geor', 'Central Kanuri (Arabic script)': 'knc_Arab', 'Central Kanuri (Latin script)': 'knc_Latn', 'Kazakh': 'kaz_Cyrl', 'Kabiyè': 'kbp_Latn', 'Kabuverdianu': 'kea_Latn', 'Khmer': 'khm_Khmr', 'Kikuyu': 'kik_Latn', 'Kinyarwanda': 'kin_Latn', 'Kyrgyz': 'kir_Cyrl', 'Kimbundu': 'kmb_Latn', 'Northern Kurdish': 'kmr_Latn', 'Kikongo': 'kon_Latn', 'Korean': 'kor_Hang', 'Lao': 'lao_Laoo', 'Ligurian': 'lij_Latn', 'Limburgish': 'lim_Latn', 'Lingala': 'lin_Latn', 'Lithuanian': 'lit_Latn', 'Lombard': 'lmo_Latn', 'Latgalian': 'ltg_Latn', 'Luxembourgish': 'ltz_Latn', 'Luba-Kasai': 'lua_Latn', 'Ganda': 'lug_Latn', 'Luo': 'luo_Latn', 'Mizo': 'lus_Latn', 'Standard Latvian': 'lvs_Latn', 'Magahi': 'mag_Deva', 'Maithili': 'mai_Deva', 'Malayalam': 'mal_Mlym', 'Marathi': 'mar_Deva', 'Minangkabau (Arabic script)': 'min_Arab', 'Minangkabau (Latin script)': 'min_Latn', 'Macedonian': 'mkd_Cyrl', 'Plateau Malagasy': 'plt_Latn', 'Maltese': 'mlt_Latn', 'Meitei (Bengali script)': 'mni_Beng', 'Halh Mongolian': 'khk_Cyrl', 'Mossi': 'mos_Latn', 'Maori': 'mri_Latn', 'Burmese': 'mya_Mymr', 'Dutch': 'nld_Latn', 'Norwegian Nynorsk': 'nno_Latn', 'Norwegian Bokmål': 'nob_Latn', 'Nepali': 'npi_Deva', 'Northern Sotho': 'nso_Latn', 'Nuer': 'nus_Latn', 'Nyanja': 'nya_Latn', 'Occitan': 'oci_Latn', 'West Central Oromo': 'gaz_Latn', 'Odia': 'ory_Orya', 'Pangasinan': 'pag_Latn', 'Eastern Panjabi': 'pan_Guru', 'Papiamento': 'pap_Latn', 'Western Persian': 'pes_Arab', 'Polish': 'pol_Latn', 'Portuguese': 'por_Latn', 'Dari': 'prs_Arab', 'Southern Pashto': 'pbt_Arab', 'Ayacucho Quechua': 'quy_Latn', 'Romanian': 'ron_Latn', 'Rundi': 'run_Latn', 'Russian': 'rus_Cyrl', 'Sango': 'sag_Latn', 'Sanskrit': 'san_Deva', 'Santali': 'sat_Olck', 'Sicilian': 'scn_Latn', 'Shan': 'shn_Mymr', 'Sinhala': 'sin_Sinh', 'Slovak': 'slk_Latn', 'Slovenian': 'slv_Latn', 'Samoan': 'smo_Latn', 'Shona': 'sna_Latn', 'Sindhi': 'snd_Arab', 'Somali': 'som_Latn', 'Southern Sotho': 'sot_Latn', 'Spanish': 'spa_Latn', 'Tosk Albanian': 'als_Latn', 'Sardinian': 'srd_Latn', 'Serbian': 'srp_Cyrl', 'Swati': 'ssw_Latn', 'Sundanese': 'sun_Latn', 'Swedish': 'swe_Latn', 'Swahili': 'swh_Latn', 'Silesian': 'szl_Latn', 'Tamil': 'tam_Taml', 'Tatar': 'tat_Cyrl', 'Telugu': 'tel_Telu', 'Tajik': 'tgk_Cyrl', 'Tagalog': 'tgl_Latn', 'Thai': 'tha_Thai', 'Tigrinya': 'tir_Ethi', 'Tamasheq (Latin script)': 'taq_Latn', 'Tamasheq (Tifinagh script)': 'taq_Tfng', 'Tok Pisin': 'tpi_Latn', 'Tswana': 'tsn_Latn', 'Tsonga': 'tso_Latn', 'Turkmen': 'tuk_Latn', 'Tumbuka': 'tum_Latn', 'Turkish': 'tur_Latn', 'Twi': 'twi_Latn', 'Central Atlas Tamazight': 'tzm_Tfng', 'Uyghur': 'uig_Arab', 'Ukrainian': 'ukr_Cyrl', 'Umbundu': 'umb_Latn', 'Urdu': 'urd_Arab', 'Northern Uzbek': 'uzn_Latn', 'Venetian': 'vec_Latn', 'Vietnamese': 'vie_Latn', 'Waray': 'war_Latn', 'Wolof': 'wol_Latn', 'Xhosa': 'xho_Latn', 'Eastern Yiddish': 'ydd_Hebr', 'Yoruba': 'yor_Latn', 'Yue Chinese': 'yue_Hant', 'Chinese (Simplified)': 'zho_Hans', 'Chinese (Traditional)': 'zho_Hant', 'Standard Malay': 'zsm_Latn', 'Zulu': 'zul_Latn'}
LID = None
model_dict = None
nltk_download = None

# Start editing here
def load_models(): #TODO: only use one model, that is defined in the config
    model_name_dict = {
        '0.6B': 'facebook/nllb-200-distilled-600M',
        '1.3B': 'facebook/nllb-200-distilled-1.3B',
        '3.3B': 'facebook/nllb-200-3.3B',
    }

    model_dict = {}
    for call_name, real_name in model_name_dict.items():
        model = AutoModelForSeq2SeqLM.from_pretrained(real_name, torch_dtype=torch.bfloat16).to(device) #TODO:Device Map?
        tokenizer = AutoTokenizer.from_pretrained(real_name)
        model_dict[call_name+'_model'] = model
        model_dict[call_name+'_tokenizer'] = tokenizer

    return model_dict

def detect_language(text): #TODO: no extra function needed
    predictions = LID.predict(text)
    detected_lang_code = predictions[0][0].replace("__label__", "")
    return detected_lang_code

def translation(model_name, sentence_mode, selection_mode, source, target, text):
    start_time = time.time()

    # Determine the source language
    if selection_mode == "Auto-detect": #TODO: "auto" should be a possible source parameter
        detected_lang_code = detect_language(text)
        flores_source_code = detected_lang_code
        source_code = flores_source_code #TODO: Superflues, refactor the complete code block
    else:
        if source == "Auto-detect":  # Make sure we don't use "Auto-detect" as a key
            return {'error': "Source language cannot be 'Auto-detect' when selection mode is manual."}
        source_code = flores_codes.get(source)
        if not source_code:
            return {'error': f"Source language {source} not found in flores_codes."}

    target_code = flores_codes[target]
    model = model_dict[model_name + '_model']
    tokenizer = model_dict[model_name + '_tokenizer']

    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_code, tgt_lang=target_code, device=device) #TODO:Device Problem, this above

    #TODO: Make Sentence-wise the only mode
    if sentence_mode == "Sentence-wise":
        sentences = sent_tokenize(text)
        translated_sentences = []
        for sentence in sentences:
            translated_sentence = translator(sentence, max_length=400)[0]['translation_text']
            translated_sentences.append(translated_sentence)
        output = ' '.join(translated_sentences)
    else:
        output = translator(text, max_length=400)[0]['translation_text']

    end_time = time.time()

    result = {
        'inference_time': end_time - start_time,
        'source_language': source_code,
        'target_language': target_code,
        'result': output
    }

    return result
# Finish editing here

### AI service section
@app.route("/v1/{url_part}".format(url_part = config["functions"][0]["url_part"]), methods=["POST"])
def function_translator(): #TODO: don't use hard coded name here.
# Start editing here
    data = request.json
    model_name = "3.3B" #TODO: Get model always from config
    selection_mode = "Manually select"
    sentence_mode = "Sentence-wise"
    device_map = data.get("device_map", "")
    source = data.get("source_language", "")
    target = data.get("target_language", "")
    text = data.get("text", "")

    result = translation(model_name, sentence_mode, selection_mode, source, target, text)

    id = str(uuid.uuid4())
    status = "Processed"
    response_text = str(result["result"])
    estimated_response_at = get_estimated_response_at(1)

    response = {
        "id": id,
        "estimated_response_at": estimated_response_at,
        "progress": 100,
        "status": status,
        "response": response_text,
        "file_attachements": list()
    }

    return jsonify(response), 201
# Finish editing here

@app.route("/v1/health-check", methods=["GET"])
def health_check(): #TODO: I don't get this, how is this checking anything? Do you want something like status here?
    return {
        "status": "Ok"
    }, 200

@app.route("/v1/setup", methods=["POST"])
def setup():
# Start editing here
    global LID
    global nltk_download
    global model_dict
    data = request.json
    force_setup = data.get("force_setup", False)

    if not os.path.isfile(file) or force_setup:
        response = requests.get(url)
        open(file, "wb").write(response.content)

    if LID == None:
        LID = fasttext.load_model(file)
    if nltk_download == None:
        nltk.download('punkt')
        nltk_download = True
    if model_dict == None:
        model_dict = load_models()
# Finish editing here
    return {
        "setup": "Performed"
    }, 201

#TODO: The api-testcall is missing. import request ...