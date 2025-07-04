import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser
import glob

config = configparser.ConfigParser()
config.read('config.ini')

class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        if isinstance(module_outputs, tuple):
            self.out = module_outputs[0]
        else:
            self.out = module_outputs

def load_model(model_family: str, model_size: str, model_type: str, device: str):
    model_path = os.path.join(
        config[model_family]['weights_directory'],
        config[model_family][f'{model_size}_{model_type}_subdir']
    )
    try:
        # load tokenizer & model
        if model_family == 'Llama2':
            tokenizer = LlamaTokenizer.from_pretrained(str(model_path))
            model = LlamaForCausalLM.from_pretrained(str(model_path))
            tokenizer.bos_token = '<s>'
        else:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(str(model_path))

        # decide dtype based on family & device
        if device == 'cpu':
            dtype = t.float32
        else:
            # Fix: Only use bfloat16 if supported
            if model_family.lower() == "gemma2" and t.cuda.is_available() and t.cuda.is_bf16_supported():
                dtype = t.bfloat16
            else:
                dtype = t.float16

        model = model.to(device=device, dtype=dtype)

        # abstract layer container for hook registration
        if hasattr(model.model, 'layers'):
            model._custom_layers = model.model.layers
        elif hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'layers'):
            model._custom_layers = model.model.transformer.layers
        else:
            raise ValueError("Unknown transformer layer structure")

        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_statements(dataset_name):
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    return dataset['statement'].tolist()

def get_acts(statements, tokenizer, model, layers, device, model_family):
    attn_hooks = {}
    mlp_hooks = {}
    handles = []

    for l in layers:
        attn_hook = Hook()
        mlp_hook = Hook()
        # switch attribute for Gemma families
        layer_module = model._custom_layers[l]
        if 'gemma' in model_family.lower():
            handles.append(layer_module.self_attention.register_forward_hook(attn_hook))
        else:
            handles.append(layer_module.self_attn.register_forward_hook(attn_hook))
        handles.append(layer_module.mlp.register_forward_hook(mlp_hook))
        attn_hooks[l] = attn_hook
        mlp_hooks[l] = mlp_hook

    acts = {2 * l: [] for l in layers}
    acts.update({2 * l + 1: [] for l in layers})

    for statement in tqdm(statements):
        input_ids = tokenizer.encode(statement, return_tensors="pt", add_special_tokens=True).to(device)
        model(input_ids)
        for l in layers:
            # grab last-token activation
            acts[2 * l].append(attn_hooks[l].out[0, -1])
            acts[2 * l + 1].append(mlp_hooks[l].out[0, -1])

    for k in acts:
        acts[k] = t.stack(acts[k]).float()

    for handle in handles:
        handle.remove()

    return acts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model_family", default="Llama3", help="Model family: Llama2, Llama3, Gemma, Gemma2, Mistral.")
    parser.add_argument("--model_size", default="8B", help="Model size for Llama3: 8B or 70B.")
    parser.add_argument("--model_type", default="base", help="Model type: base or chat.")
    parser.add_argument("--layers", nargs='+', help="Real layer indices to extract from.")
    parser.add_argument("--datasets", nargs='+', help="Dataset names (no .csv).")
    parser.add_argument("--output_dir", default="acts", help="Directory to save activations.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    datasets = args.datasets
    if datasets == ['all_topic_specific']:
        datasets = [
            'cities', 'sp_en_trans', 'inventors', 'animal_class', 'element_symb', 'facts',
            'neg_cities', 'neg_sp_en_trans', 'neg_inventors', 'neg_animal_class', 'neg_element_symb', 'neg_facts',
            'cities_conj', 'sp_en_trans_conj', 'inventors_conj', 'animal_class_conj', 'element_symb_conj', 'facts_conj',
            'cities_disj', 'sp_en_trans_disj', 'inventors_disj', 'animal_class_disj', 'element_symb_disj', 'facts_disj',
            'larger_than', 'smaller_than', 'cities_de', 'neg_cities_de', 'sp_en_trans_de', 'neg_sp_en_trans_de',
            'inventors_de', 'neg_inventors_de', 'animal_class_de', 'neg_animal_class_de', 'element_symb_de',
            'neg_element_symb_de', 'facts_de', 'neg_facts_de'
        ]
    if datasets == ['all']:
        datasets = []
        for file_path in glob.glob('datasets/**/*.csv', recursive=True):
            dataset_name = os.path.relpath(file_path, 'datasets').replace('.csv', '')
            datasets.append(dataset_name)

    t.set_grad_enabled(False)
    tokenizer, model = load_model(
        args.model_family, args.model_size, args.model_type, args.device
    )

    for dataset in datasets:
        statements = load_statements(dataset)
        layers = [int(x) for x in args.layers]
        if layers == [-1]:
            layers = list(range(len(model._custom_layers)))
        save_dir = f"{args.output_dir}/{args.model_family}/{args.model_size}/{args.model_type}/{dataset}/"
        os.makedirs(save_dir, exist_ok=True)

        for idx in range(0, len(statements), 25):
            batch = statements[idx:idx + 25]
            acts = get_acts(batch, tokenizer, model, layers, args.device, args.model_family)
            for pseudo_layer, act in acts.items():
                for i in range(act.size(0)):
                    t.save(act[i], f"{save_dir}/layer_{pseudo_layer}_{idx + i}.pt")
