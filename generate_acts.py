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
    def __init__(self, name):
        self.out = None
        self.name = name

    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs

def load_model(model_family: str, model_size: str, model_type: str, device: str):
    model_path = os.path.join(config[model_family]['weights_directory'], 
                              config[model_family][f'{model_size}_{model_type}_subdir'])
    
    try:
        if model_family == 'Llama2':
            tokenizer = LlamaTokenizer.from_pretrained(str(model_path))
            model = LlamaForCausalLM.from_pretrained(str(model_path))
            tokenizer.bos_token = '<s>'
        else:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(str(model_path))
        if model_family == "Gemma2":
            #edit1 begins
            model = model.to(t.bfloat16)
        else:
            model = model.half()
            #edit1 ends
        return tokenizer, model.to(device)
    except Exception as e:   
        print(f"Error loading model: {e}")
        raise

def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    return statements

def get_acts(statements, tokenizer, model, layers, device, save_dir, batch_offset):
    hooks, handles = {}, []

    for layer in layers:
        layer_module = model.model.layers[layer]

        hook_attn = Hook(f"attn_{layer}")
        handles.append(layer_module.self_attn.register_forward_hook(hook_attn))
        hooks[f"attn_{layer}"] = hook_attn

        if hasattr(layer_module, "mlp"):
            hook_ffn = Hook(f"ffn_{layer}")
            handles.append(layer_module.mlp.register_forward_hook(hook_ffn))
            hooks[f"ffn_{layer}"] = hook_ffn
        elif hasattr(layer_module, "post_attention"):
            hook_ffn = Hook(f"ffn_{layer}")
            handles.append(layer_module.post_attention.register_forward_hook(hook_ffn))
            hooks[f"ffn_{layer}"] = hook_ffn
        else:
            raise ValueError(f"Cannot find MLP or post_attention in layer {layer}")

    acts = {name: [] for name in hooks}

    for statement in tqdm(statements):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        model(input_ids)

        for name, hook in hooks.items():
            acts[name].append(hook.out[0][-1])

    for name, tensors in acts.items():
        layer_type, block_id = name.split("_")
        block_id = int(block_id)
        file_layer_idx = 2 * block_id if layer_type == "attn" else 2 * block_id + 1
    
        # Fix: Ensure tensors are stacked consistently by taking only the last token
        stacked = t.stack([x[-1] for x in tensors]).float()

        file_path = os.path.join(save_dir, f"layer_{file_layer_idx}_{batch_offset}.pt")
        t.save(stacked, file_path)


    for handle in handles:
        handle.remove()
        
if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model_family", default="Llama3", help="Model family to use. Options are Llama2, Llama3, Gemma, Gemma2 or Mistral.")
    parser.add_argument("--model_size", default="8B",
                        help="Size of the model to use. Options for Llama3 are 8B or 70B")
    parser.add_argument("--model_type", default="base", help="Whether to choose base or chat model. Options are base or chat.")
    parser.add_argument("--layers", nargs='+', 
                        help="Layers to save embeddings from.")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="acts",
                        help="Directory to save activations to.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    datasets = args.datasets
    if datasets == ['all_topic_specific']:
        datasets = ['cities', 'sp_en_trans', 'inventors', 'animal_class', 'element_symb', 'facts',
                    'neg_cities', 'neg_sp_en_trans', 'neg_inventors', 'neg_animal_class', 'neg_element_symb', 'neg_facts',
                    'cities_conj', 'sp_en_trans_conj', 'inventors_conj', 'animal_class_conj', 'element_symb_conj', 'facts_conj',
                    'cities_disj', 'sp_en_trans_disj', 'inventors_disj', 'animal_class_disj', 'element_symb_disj', 'facts_disj',
                    'larger_than', 'smaller_than', "cities_de", "neg_cities_de", "sp_en_trans_de", "neg_sp_en_trans_de", "inventors_de", "neg_inventors_de", "animal_class_de",
                  "neg_animal_class_de", "element_symb_de", "neg_element_symb_de", "facts_de", "neg_facts_de"]
    if datasets == ['all']:
        datasets = []
        for file_path in glob.glob('datasets/**/*.csv', recursive=True):
            dataset_name = os.path.relpath(file_path, 'datasets').replace('.csv', '')
            datasets.append(dataset_name)

    t.set_grad_enabled(False)
    tokenizer, model = load_model(args.model_family, args.model_size, args.model_type, args.device)
    
    for dataset in datasets:
        statements = load_statements(dataset)
        layers = [int(layer) for layer in args.layers]
        if layers == [-1]:
            layers = list(range(len(model.model.layers)))
        save_dir = f"{args.output_dir}/{args.model_family}/{args.model_size}/{args.model_type}/{dataset}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(0, len(statements), 25):
            get_acts(
                statements=statements[idx:idx + 25],
                tokenizer=tokenizer,
                model=model,
                layers=layers,
                device=args.device,
                save_dir=save_dir,
                batch_offset=idx
            )
