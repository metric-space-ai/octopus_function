from datetime import datetime, timedelta
import threading

import requests
import os
import torch
import gradio as gr
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import fasttext
import nltk
from nltk.tokenize import sent_tokenize

class Translator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.file = "lid218e.bin"
        self.flores_codes={'Acehnese (Arabic script)': 'ace_Arab', 'Acehnese (Latin script)': 'ace_Latn', 'Mesopotamian Arabic': 'acm_Arab', 'Ta’izzi-Adeni Arabic': 'acq_Arab', 'Tunisian Arabic': 'aeb_Arab', 'Afrikaans': 'afr_Latn', 'South Levantine Arabic': 'ajp_Arab', 'Akan': 'aka_Latn', 'Amharic': 'amh_Ethi', 'North Levantine Arabic': 'apc_Arab', 'Modern Standard Arabic': 'arb_Arab', 'Modern Standard Arabic (Romanized)': 'arb_Latn', 'Najdi Arabic': 'ars_Arab', 'Moroccan Arabic': 'ary_Arab', 'Egyptian Arabic': 'arz_Arab', 'Assamese': 'asm_Beng', 'Asturian': 'ast_Latn', 'Awadhi': 'awa_Deva', 'Central Aymara': 'ayr_Latn', 'South Azerbaijani': 'azb_Arab', 'North Azerbaijani': 'azj_Latn', 'Bashkir': 'bak_Cyrl', 'Bambara': 'bam_Latn', 'Balinese': 'ban_Latn', 'Belarusian': 'bel_Cyrl', 'Bemba': 'bem_Latn', 'Bengali': 'ben_Beng', 'Bhojpuri': 'bho_Deva', 'Banjar (Arabic script)': 'bjn_Arab', 'Banjar (Latin script)': 'bjn_Latn', 'Standard Tibetan': 'bod_Tibt', 'Bosnian': 'bos_Latn', 'Buginese': 'bug_Latn', 'Bulgarian': 'bul_Cyrl', 'Catalan': 'cat_Latn', 'Cebuano': 'ceb_Latn', 'Czech': 'ces_Latn', 'Chokwe': 'cjk_Latn', 'Central Kurdish': 'ckb_Arab', 'Crimean Tatar': 'crh_Latn', 'Welsh': 'cym_Latn', 'Danish': 'dan_Latn', 'German': 'deu_Latn', 'Southwestern Dinka': 'dik_Latn', 'Dyula': 'dyu_Latn', 'Dzongkha': 'dzo_Tibt', 'Greek': 'ell_Grek', 'English': 'eng_Latn', 'Esperanto': 'epo_Latn', 'Estonian': 'est_Latn', 'Basque': 'eus_Latn', 'Ewe': 'ewe_Latn', 'Faroese': 'fao_Latn', 'Fijian': 'fij_Latn', 'Finnish': 'fin_Latn', 'Fon': 'fon_Latn', 'French': 'fra_Latn', 'Friulian': 'fur_Latn', 'Nigerian Fulfulde': 'fuv_Latn', 'Scottish Gaelic': 'gla_Latn', 'Irish': 'gle_Latn', 'Galician': 'glg_Latn', 'Guarani': 'grn_Latn', 'Gujarati': 'guj_Gujr', 'Haitian Creole': 'hat_Latn', 'Hausa': 'hau_Latn', 'Hebrew': 'heb_Hebr', 'Hindi': 'hin_Deva', 'Chhattisgarhi': 'hne_Deva', 'Croatian': 'hrv_Latn', 'Hungarian': 'hun_Latn', 'Armenian': 'hye_Armn', 'Igbo': 'ibo_Latn', 'Ilocano': 'ilo_Latn', 'Indonesian': 'ind_Latn', 'Icelandic': 'isl_Latn', 'Italian': 'ita_Latn', 'Javanese': 'jav_Latn', 'Japanese': 'jpn_Jpan', 'Kabyle': 'kab_Latn', 'Jingpho': 'kac_Latn', 'Kamba': 'kam_Latn', 'Kannada': 'kan_Knda', 'Kashmiri (Arabic script)': 'kas_Arab', 'Kashmiri (Devanagari script)': 'kas_Deva', 'Georgian': 'kat_Geor', 'Central Kanuri (Arabic script)': 'knc_Arab', 'Central Kanuri (Latin script)': 'knc_Latn', 'Kazakh': 'kaz_Cyrl', 'Kabiyè': 'kbp_Latn', 'Kabuverdianu': 'kea_Latn', 'Khmer': 'khm_Khmr', 'Kikuyu': 'kik_Latn', 'Kinyarwanda': 'kin_Latn', 'Kyrgyz': 'kir_Cyrl', 'Kimbundu': 'kmb_Latn', 'Northern Kurdish': 'kmr_Latn', 'Kikongo': 'kon_Latn', 'Korean': 'kor_Hang', 'Lao': 'lao_Laoo', 'Ligurian': 'lij_Latn', 'Limburgish': 'lim_Latn', 'Lingala': 'lin_Latn', 'Lithuanian': 'lit_Latn', 'Lombard': 'lmo_Latn', 'Latgalian': 'ltg_Latn', 'Luxembourgish': 'ltz_Latn', 'Luba-Kasai': 'lua_Latn', 'Ganda': 'lug_Latn', 'Luo': 'luo_Latn', 'Mizo': 'lus_Latn', 'Standard Latvian': 'lvs_Latn', 'Magahi': 'mag_Deva', 'Maithili': 'mai_Deva', 'Malayalam': 'mal_Mlym', 'Marathi': 'mar_Deva', 'Minangkabau (Arabic script)': 'min_Arab', 'Minangkabau (Latin script)': 'min_Latn', 'Macedonian': 'mkd_Cyrl', 'Plateau Malagasy': 'plt_Latn', 'Maltese': 'mlt_Latn', 'Meitei (Bengali script)': 'mni_Beng', 'Halh Mongolian': 'khk_Cyrl', 'Mossi': 'mos_Latn', 'Maori': 'mri_Latn', 'Burmese': 'mya_Mymr', 'Dutch': 'nld_Latn', 'Norwegian Nynorsk': 'nno_Latn', 'Norwegian Bokmål': 'nob_Latn', 'Nepali': 'npi_Deva', 'Northern Sotho': 'nso_Latn', 'Nuer': 'nus_Latn', 'Nyanja': 'nya_Latn', 'Occitan': 'oci_Latn', 'West Central Oromo': 'gaz_Latn', 'Odia': 'ory_Orya', 'Pangasinan': 'pag_Latn', 'Eastern Panjabi': 'pan_Guru', 'Papiamento': 'pap_Latn', 'Western Persian': 'pes_Arab', 'Polish': 'pol_Latn', 'Portuguese': 'por_Latn', 'Dari': 'prs_Arab', 'Southern Pashto': 'pbt_Arab', 'Ayacucho Quechua': 'quy_Latn', 'Romanian': 'ron_Latn', 'Rundi': 'run_Latn', 'Russian': 'rus_Cyrl', 'Sango': 'sag_Latn', 'Sanskrit': 'san_Deva', 'Santali': 'sat_Olck', 'Sicilian': 'scn_Latn', 'Shan': 'shn_Mymr', 'Sinhala': 'sin_Sinh', 'Slovak': 'slk_Latn', 'Slovenian': 'slv_Latn', 'Samoan': 'smo_Latn', 'Shona': 'sna_Latn', 'Sindhi': 'snd_Arab', 'Somali': 'som_Latn', 'Southern Sotho': 'sot_Latn', 'Spanish': 'spa_Latn', 'Tosk Albanian': 'als_Latn', 'Sardinian': 'srd_Latn', 'Serbian': 'srp_Cyrl', 'Swati': 'ssw_Latn', 'Sundanese': 'sun_Latn', 'Swedish': 'swe_Latn', 'Swahili': 'swh_Latn', 'Silesian': 'szl_Latn', 'Tamil': 'tam_Taml', 'Tatar': 'tat_Cyrl', 'Telugu': 'tel_Telu', 'Tajik': 'tgk_Cyrl', 'Tagalog': 'tgl_Latn', 'Thai': 'tha_Thai', 'Tigrinya': 'tir_Ethi', 'Tamasheq (Latin script)': 'taq_Latn', 'Tamasheq (Tifinagh script)': 'taq_Tfng', 'Tok Pisin': 'tpi_Latn', 'Tswana': 'tsn_Latn', 'Tsonga': 'tso_Latn', 'Turkmen': 'tuk_Latn', 'Tumbuka': 'tum_Latn', 'Turkish': 'tur_Latn', 'Twi': 'twi_Latn', 'Central Atlas Tamazight': 'tzm_Tfng', 'Uyghur': 'uig_Arab', 'Ukrainian': 'ukr_Cyrl', 'Umbundu': 'umb_Latn', 'Urdu': 'urd_Arab', 'Northern Uzbek': 'uzn_Latn', 'Venetian': 'vec_Latn', 'Vietnamese': 'vie_Latn', 'Waray': 'war_Latn', 'Wolof': 'wol_Latn', 'Xhosa': 'xho_Latn', 'Eastern Yiddish': 'ydd_Hebr', 'Yoruba': 'yor_Latn', 'Yue Chinese': 'yue_Hant', 'Chinese (Simplified)': 'zho_Hans', 'Chinese (Traditional)': 'zho_Hant', 'Standard Malay': 'zsm_Latn', 'Zulu': 'zul_Latn'}
        self.LID = None
        self.model_dict = None
        self.nltk_download = None
        self.results = {}

    def get_estimated_response_at(self, seconds: int) -> str:
        estimated_response_at = str(datetime.now() + timedelta(seconds))
        estimated_response_at = estimated_response_at + "Z"
        estimated_response_at = estimated_response_at.replace(" ", "T")

        return estimated_response_at

    def function_translator_get_response(self, id):
        return self.results[id]

    def setup(self):
        URL = str("https://dl.fbaipublicfiles.com/nllb/lid/" + self.file)
        response = requests.get(URL)
        open(self.file, "wb").write(response.content)

    def setup_condition(self):
        if not os.path.isfile(self.file):
            return True

        return False

    def function_translator_start_thread(self, model_name, sentence_mode, selection_mode, source, target, text, id):
        thread = threading.Thread(target=self.translation, args=(model_name, sentence_mode, selection_mode, source, target, text, id,))
        thread.start()

    def function_translator_store_response(self, id, response):
        self.results[id] = response

    def warmup(self):
        if self.LID == None:
            self.LID = fasttext.load_model("lid218e.bin")
        if self.nltk_download == None:
            nltk.download('punkt')
            self.nltk_download = True
        if self.model_dict == None:
            self.model_dict = self.load_models()

    def warmup_condition(self):
        if self.LID == None or self.nltk_download == None or self.model_dict == None:
            return True

        return False

    def load_models(self):
        model_name_dict = {
            '0.6B': 'facebook/nllb-200-distilled-600M',
            '1.3B': 'facebook/nllb-200-distilled-1.3B',
            '3.3B': 'facebook/nllb-200-3.3B',
        }

        model_dict = {}
        for call_name, real_name in model_name_dict.items():
            model = AutoModelForSeq2SeqLM.from_pretrained(real_name, torch_dtype=torch.bfloat16).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(real_name)
            model_dict[call_name+'_model'] = model
            model_dict[call_name+'_tokenizer'] = tokenizer

        return model_dict

    def detect_language(self, text):
        predictions = self.LID.predict(text)
        detected_lang_code = predictions[0][0].replace("__label__", "")
        return detected_lang_code

    def translation(self, model_name, sentence_mode, selection_mode, source, target, text, id):
        self.warmup()

        start_time = time.time()

        # Determine the source language
        if selection_mode == "Auto-detect":
            detected_lang_code = self.detect_language(text)
            flores_source_code = detected_lang_code
            source_code = flores_source_code
        else:
            if source == "Auto-detect":  # Make sure we don't use "Auto-detect" as a key
                return {'error': "Source language cannot be 'Auto-detect' when selection mode is manual."}
            source_code = self.flores_codes.get(source)
            if not source_code:
                return {'error': f"Source language {source} not found in flores_codes."}

        target_code = self.flores_codes[target]
        model = self.model_dict[model_name + '_model']
        tokenizer = self.model_dict[model_name + '_tokenizer']

        translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_code, tgt_lang=target_code, device=self.device)

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

        response = self.function_translator_get_response(id)
        estimated_response_at = self.get_estimated_response_at(0)
        response["estimated_response_at"] = estimated_response_at
        response["progress"] = 100
        response["response"] = str(result["result"]),
        response["status"] = "Processed"
        self.function_translator_store_response(id, response)

        return result
