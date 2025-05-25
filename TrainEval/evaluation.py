from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

import torch
state_dict = torch.load('nllb_saTOhi_finetuned4.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

import pandas as pd
test_json = pd.read_json("test_file.json", lines=True)
test_df = pd.json_normalize(test_json["translation"])

import sacrebleu

def get_metrics(pred, ref):
    bleu = sacrebleu.sentence_bleu(pred, [ref]).score
    chrf = sacrebleu.sentence_chrf(pred, [ref]).score
    return bleu, chrf


gpu_id=int(input("Enter GPU ID: "))
try:
    torch.cuda.set_device(gpu_id)
except Exception as e:
    print(e)

if torch.cuda.current_device()!=gpu_id:
    print("CURRENT GPU: ", torch.cuda.current_device())
    exit(1)





from tqdm import tqdm  # add this import

results = []
for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
    src = row['sa']
    ref = row['hi']
    
    inputs = tokenizer(src, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=256)
    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    
    bleu, chrf = get_metrics(pred, ref)
    
    results.append({
        'sa': src,
        'hi_ref': ref,
        'hi_pred': pred,
        'BLEU': bleu,
        'CHRF++': chrf
    })
    
    
df_results = pd.DataFrame(results)
df_results.to_csv('sa_hi_eval4.csv', index=False)

