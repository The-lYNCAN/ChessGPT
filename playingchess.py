from decoderArchitecture import ChessGPT
import torch 
import json
from chess_game import main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('token_to_id', 'r') as file:
    # 2. Parse the JSON data back into a Python object
    token_to_id = json.load(file)
with open('id_to_token', 'r') as file:
    # 2. Parse the JSON data back into a Python object
    id_to_token = json.load(file)

model = ChessGPT(vocab_size=len(token_to_id), d_model=512, n_heads=8, d_ff=2048, n_layers=10, max_seq_len=250).to(device)
model.load_state_dict(torch.load('checkpoint_epoch_15.pt', map_location=device))

def generate(model, token_to_id, id_to_token, prompt_ids, max_new_tokens, device, temperature=1.0):
    model.eval()
    ids = prompt_ids.copy()

    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([ids], dtype=torch.long).to(device)  # (1, T)

        with torch.no_grad():
            logits = model(input_tensor)  # (1, T, vocab_size)

        # only care about the prediction for the NEXT token — last position
        next_token_logits = logits[0, -1, :] / temperature  # (vocab_size,)

        probs = torch.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        ids.append(next_id)

    return ids

pred = generate(model, token_to_id, id_to_token, [token_to_id['e2e4']], 1, device, temperature=0.2)
print(pred)
main()