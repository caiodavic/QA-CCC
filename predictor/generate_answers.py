import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM,LlamaTokenizer, GenerationConfig
from search_engine import SearchEngine


PROMPT_CONTEXT = """Abaixo está uma instrução que descreve uma tarefa sobre o curso de Ciência da Computação na Universidade Federal de Campina Grande. A instrução é sobre: [CATEGORY], juntamente com uma entrada que fornece mais contexto. As informações em ### Contexto devem ser utilizadas para resposta. Escreva uma resposta em linguagem natural e em Português que complete adequadamente o pedido.\n\n
### Instrução:\n
[QUESTION]\n\n
### Contexto: \n
[CONTEXT]\n\n
### Resposta: \n"""

PROMPT_NO_CONTEXT = """"Abaixo está uma instrução que descreve uma tarefa sobre o curso de Ciência da Computação na Universidade Federal de Campina Grande.Escreva uma resposta que complete adequadamente o pedido. Escreva uma resposta em linguagem natural e em Português que complete adequadamento o pedido.\n\n
### Instrução:\n
[QUESTION]\n\n
### Resposta: \n
"""

class Generator():
    def __init__(self,model_path:str=None,search_engine:SearchEngine=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            "decapoda-research/llama-13b-hf",
            load_in_8bit=True,
            device_map="auto",
        )
        self.tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-13b-hf")

        if model_path:
            self.model = PeftModel.from_pretrained(self.model, model_path)
            print('Modelo tunado carregado com sucesso')
        if search_engine:
            self.search_engine = search_engine
        else:
            self.search_engine = SearchEngine()

        self.generation_config = GenerationConfig(
            temperature=0.2,
        )
    
    def formatting_func(self,question:str,context:str=None,category:str=None) -> str:
        if context:
            return PROMPT_CONTEXT.replace("[QUESTION]",question).replace("[CONTEXT]",context).replace("[CATEGORY]",category)    
        else: 
            return PROMPT_NO_CONTEXT.replace("[QUESTION]",question)

    def search_in_engine(self,question:str) -> dict:
        return self.search_engine.search_context(question)

    def generate_prompt(self,question,search_engine:bool=True) -> str:
        if search_engine:
            dict_context = self.search_in_engine(question)
        if search_engine and dict_context:
            if 'context' in dict_context:
                return self.formatting_func(question,dict_context['context'],dict_context['category'])
            return self.formatting_func(question,dict_context['answer'],dict_context['category'])
        else:
            print('Sem contexto')
            return self.formatting_func(question) 
    
    def generate_answer(self,question:str, search_engine:bool=True) -> str:
        #prompt = self.generate_prompt(question,search_engine=search_engine)
        inputs = self.tokenizer(question, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_output = self.model.generate(
        input_ids=input_ids,
        generation_config=self.generation_config,
        return_dict_in_generate=True,
        output_scores=True,
            max_new_tokens=512
    )
        for s in generation_output.sequences:
            output = self.tokenizer.decode(s)
            result = output.split("### Response:")[1].strip()
            return result
        return output