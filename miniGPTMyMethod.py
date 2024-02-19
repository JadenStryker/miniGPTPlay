import torch
import torch.nn as nn
from torch.nn import functional as F
import json

# My Hyperparameters

batch_size = 32 # proccess 32 examples at a time
block_size = 8 # effective context length, not really relevant for bi grams? 
max_iters = 5000 # number of times to run a batch
eval_iter = 300 # ! number of times before displaying recent performance results? seems wrong
learning_rate = 1e-3
n_embed = 32
n_layer = 4
n_head = 4 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
droupout = .2
torch.manual_seed(42)

with open('input.txt', 'r') as file:
    text_data = file.read()
    
chars = sorted(list(set(text_data)))
vocab_size = len(chars) # should be 65 for the shakspere example

# writer the char -> int encoder, and int -> char decoder as lambda functions
char_int = { char : integer for integer, char in enumerate(chars)}
int_char = { integer : char for integer, char in enumerate(chars)}
encode = lambda str: [char_int[char] for char in str]
decode = lambda int_list: [int_char[integer] for integer in int_list]

data = torch.tensor(encode(text_data), dtype = torch.long) #turn the shakspere text into a vector using the encoder for each char
split_length = int(.9 * len(data))
train_data = data[:split_length]
test_data = data[split_length:]


def get_batch(split):
    data = train_data if split == 'train' else test_data
    random_indexs = torch.randint(len(data) - block_size, (batch_size,)) # gather a random integer block size + 1 less than total, get batch_size of these indexs
    fill_x = torch.stack([data[i:i+block_size] for i in random_indexs])
    fill_y = torch.stack([data[i+1:i+block_size+1] for i in random_indexs])
    x, y = fill_x.to(device), fill_y.to(device)
    return x,y

@torch.no_grad() # anything happens here will not get backward.
def estimate_loss(): # take 300 iterations, average it, report it
    out = {}
    model.eval() # * rather this has to do with batch norm and drop out and so on 
    for split in ['train', 'val']: # test train loss, test validation loss, hasnt seen val set yet. prevents from evaluating hyper parametrs on test set
        losses = torch.zeros(eval_iter) # see
        for k in range(eval_iter):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module): # how do encodings come into here? 
    """ one head of self attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size))) # not a learnable parameter so becomes a register buffer. i imagine this has to do with gpu optimizaiton
        self.droupout = nn.Dropout(droupout)
    def forward(self, x): # does the encoder / translatin model. just get added above on x? and then shift triangluar matrix down. i dont know
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # compute the attention scalars
        wei = q @ k.transpose(-2,-1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # * mask to immitate next token prediction learning
        wei = F.softmax(wei, dim=-1) # * softmax it up, basically make it a distribution
        wei = self.droupout(wei)
        out = wei @ v # * this will be our new representation of the tokens, embedded in a single attention head here. (B, T, C(head_size))
        return out
    
class MultiHeadAttention(nn.Module):
    """This is the test part, reducing to head_size / n_head feels reductive, lets try just stacking and summing"""
    
    def __init__(self, num_heads, head_size): # ! head size simply means the output dimension. yes. what does concat mean in this context? 
        super().__init__()
        head_size = head_size * n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        
    def forward(self, x):
        # TODO the following stack results in a better funciton than the 
        stacked_tensors =  torch.stack([h(x) for h in self.heads], dim = -1)
        return self.proj(torch.sum(stacked_tensors, dim=-1))
        # out = torch.cat([h(x) for h in self.heads], dim = -1)
        # out = self.proj(out) # * just another feed forward. Is there a point other than adding complexity? no
        # return out # will each dimension now be of shape head_size * num_heads, is this a concat across this dimension? i doubt it would go across the time step dimension. YES
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""    
    
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(droupout)
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """This is a transformer block, it somehow integrates communication (Attention) followed by computation (FF) """
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head # ! so like if dim is 32 and 4 heads then each head is 4 dim
        self.mha = MultiHeadAttention(n_head, head_size) #! for tokens previous to our LAST / NEXT Token. why change them with a FF. this feels counter intutive. we are giving to diverse data to the attention heads. 
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.mha(self.ln1(x)) # * these are our feed forwards. the Channel token embeddings are summed
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    """ uses a token -> embedding -> self attention weighted vector -> to calculate the next step here """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) #create our super simple neuron table of logits
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # this becomes the cosine thing right? does not have to do with tokens, but rather posittion
        self.lm_head = nn.Linear(n_embed, vocab_size) # linear layer to add some complexity. Is this not the feed forward part, it is, we an just keep advancing it
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        
    def forward(self, input_ints, targets=None):
        B, T = input_ints.shape 
        tok_emb = self.token_embedding_table(input_ints) # (B, T, C) --> the index values are searched. The vector for each position 'looked up' by the index's will be of shape len(input_ints), vocab_size
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C) here we find the positional embeddings for each position in input_ints. so pos 0 gets a pos_emb and pos 1 gets its own pos. Will return a embedding for each index
        x = tok_emb + pos_emb # (B, T, C) C is the n_embed or number of channels
        # TODO here i want to stack and average all previous vectors for a generation expirement to see what the loss will be like
        x = self.blocks(x) # (B, T, C). we do multiple of these. it calculates the attention each time
        x = self.ln(x)
        logits = self.lm_head(x) # (B, T, vocab_size) #! last layer? no all layers
        
        if targets is None: #! MEANT FOR GENERATION
            loss = None
            return logits, None
        
        else: # ! MEANT FOR TRAINING
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # target is an index, logits of vocab_size strives to have position at target_index be highest.
            return None, loss
    
    def generate(self, input_ints, max_new_tokens): #max_new_tokens defined how long the genration will continue for, kind of like max_tokens on openAI api call, 
        #input_ints, is of shape (Batches x Context_length) or (B x T), T is tensor? 
        #! I am wondering, whats the point of batches here? i dont see how b comes in anyway? we just flatten everything out? 
        for _ in range(max_new_tokens):
            input_ints_slice = input_ints[:, -block_size:] # this ensures that we will not call the position embedding table with 10 elements, because then it wouldnt not be able to map it to a positional encoding. as it was not trained on that
            logits, _ = self(input_ints_slice) # calls the forward pass to generate logits
            logits = logits[:, -1, :] # becomes (b, C), we only take the last word. we get back the shape of the look up table and convert it to be the batch, last_char, and vocab_size vector
            probs = F.softmax(logits, dim = -1) # smoothes it out 
            next_input_int = torch.multinomial(probs, num_samples=1) # probably takes the highhest value, searches the embedding-bigram vector
            input_ints = torch.cat((input_ints, next_input_int), dim=1) 
        return input_ints
        
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

results = []

for iter in range(max_iters):
    if iter % eval_iter == 0:
        losses = estimate_loss()
        d = {'iter': iter, 'val_loss': losses['val'].item(), 'train_loss': losses['train'].item()}
        results.append(d)
        print(f'step: {iter}, train loss: {losses["train"]:.4f}, val_loss: {losses["val"]:.4f} :)')
    x_window, y_window = get_batch('train')
    # print(f'shape of x is: {x_window.shape}, shape of y is: {y_window.shape}')
    _, loss = model(x_window, y_window) #this is  a single forward pass, where we evaluate loss
    optimizer.zero_grad(set_to_none=True) # reset gradient
    loss.backward()
    optimizer.step()

import pandas as pd
df = pd.DataFrame(results)
df.to_csv('fun_gpt.csv')
    
    
init_context = torch.zeros((1,1), dtype = torch.long, device = device)
print("".join(decode(m.generate(init_context, 300)[0].tolist())))
print(sum(p.numel() for p in m.parameters())/1e6, 'Million parameters')
torch.save(model.state_dict(), 'bigram_language_model.pth')
def save_model_weights(model, file_path):
    model_weights = {}
    for name, param in model.named_parameters():
        model_weights[name] = param.detach().cpu().numpy().tolist()
    with open(file_path, 'w') as f:
        json.dump(model_weights, f, indent=4)
save_model_weights(model, 'model_weights.json')