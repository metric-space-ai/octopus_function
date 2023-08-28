from langchain.prompts import PromptTemplate
class PROMPT_CONFIG:
    def __init__(self):
        # ---------------------------------------------------
        # forming the LLaMA-2 prompt style
        # ---------------------------------------------------
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.DEFAULT_SYSTEM_PROMPT = """\
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        self.SYS_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.
        ​
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """

        self.INSTRUCTION = """CONTEXT:/n/n {context}/n
        ​
        Question: {question}"""

        SYSTEM_PROMPT = self.B_SYS + self.DEFAULT_SYSTEM_PROMPT + self.E_SYS
        self.prompt_template =  self.B_INST + SYSTEM_PROMPT + self.INSTRUCTION + self.E_INST

        llama_prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])
        self.chain_type_kwargs = {"prompt": llama_prompt}    