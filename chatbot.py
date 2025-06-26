import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 64
block_size = 128
max_iters = 200
lr = 3e-4
eval_iters = 100
num_embeddings = 384
num_heads = 4
num_layers = 4
dropout = 0.2

chars = ""
with open('E:/Saathvik/openwebtext/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
vocab_size = len(chars)

string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(num_embeddings, head_size, bias=False) # linear transformation for keys
        self.query = nn.Linear(num_embeddings, head_size, bias=False) # linear transformation for queries
        self.value = nn.Linear(num_embeddings, head_size, bias=False) # linear transformation for values
        # lower triangular matrix for masking future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 
        self.dropout = nn.Dropout(dropout) 
    
    def forward(self, x):
        # input x: (B, T, C) where B=batch size, T=block size, C=channels
        # output: (B, T, C) where C=head_size
        B, T, C = x.shape 
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) #@ (B, T, head_size)
        
        # compute weighted attention scores
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # scaled dot-product attention
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # type: ignore # mask future tokens
        weights = F.softmax(weights, dim=-1) # softmax to get attention weights
        weights = self.dropout(weights)
        v = self.value(x) # (B, T, head_size)
        output = weights @ v # matrix multiplication
        return output
    
# MultiHead Attention Module 
# (scaled dot-product attention (k, q, v) -> concatenate results -> linear projection)
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, num_embeddings)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)  # concatenate results from all heads
        output = self.dropout(self.projection(output))  # linear projection and dropout
        return output

# transformer block feed forward class (linear -> ReLU -> linear)
class FeedForward(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings),
            nn.ReLU(),
            nn.Linear(4 * num_embeddings, num_embeddings),
            nn.Dropout(dropout), # dropout for regularization (prevent overfitting)
        )

    def forward(self, x):
        return self.net(x)
    
# transformer block class (MultiHeadAttention -> FeedForward -> LayerNorm -> LayerNorm)
class Block(nn.Module):
    def __init__(self, num_embeddings, num_heads):
        super().__init__()
        head_size = num_embeddings // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(num_embeddings)
        self.ln1 = nn.LayerNorm(num_embeddings)
        self.ln2 = nn.LayerNorm(num_embeddings)
    
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)  # residual connection
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x     
    
# basic GPT language model
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, num_embeddings)
        self.blocks = nn.Sequential(*[Block(num_embeddings, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(num_embeddings) # final layer normalization
        self.lm_head = nn.Linear(num_embeddings, vocab_size) # final linear layer to project to vocab size
        self.apply(self.__init__weights) # initialize weights
    
    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        # B=batch size, T=time, C=channels (can be vocab size)
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_embeddings + position_embeddings  # (B, T, C)
        x = self.blocks(x)  # feed through transformer blocks
        x = self.ln_f(x) # final layer normalization
        logits = self.lm_head(x)  # project to vocab size (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape 
            
            # reshape logits and targets for cross-entropy loss
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(idx[:, -block_size:])  # get predictions
            logits = logits[:, -1, :] # (B, C) last time step
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = GPTLanguageModel(vocab_size).to(device)

with open('gpt_v1.pkl', 'rb') as f:
    model= pickle.load(f)
print("Model loaded from gpt-v1.pkl")

def chat_with_model():
    print("GPT Chatbot - Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check if user wants to quit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        try:
            # Encode user input as context
            encoded_input = encode(user_input)
            
            # Convert to tensor and add batch dimension
            context = torch.tensor(encoded_input, dtype=torch.long, device=device).unsqueeze(0)
            
            # Generate response
            print("\nBot: ", end="", flush=True)
            with torch.no_grad():
                generated_tokens = model.generate(context, max_new_tokens=200)
                generated_text = decode(generated_tokens[0].tolist())
                
                # Extract only the generated part (remove the input context)
                response = generated_text[len(user_input):]
                print(response)
                
        except KeyError as e:
            print(f"Error: Character '{e.args[0]}' not in vocabulary. Please use simpler text.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Start the interactive chat
if __name__ == "__main__":
    chat_with_model() 