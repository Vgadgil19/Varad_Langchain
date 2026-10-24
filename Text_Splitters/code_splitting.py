from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = '''
class Block(nn.Module):
  def __init__(self, n_emb, n_head):
    super().__init__()
    head_size = n_emb // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ff = Feedforward(n_emb)
    self.ln1 = nn.LayerNorm(n_emb)
    self.ln2 = nn.LayerNorm(n_emb)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ff(self.ln2(x))
    return x
model = GPTmodel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M Parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr = ler_rate)

for iter in range(max_iter):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_intervals == 0 or iter == max_iter - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")


    xb,yb = get_batches('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens = 1000)[0].tolist()))
#open('generated.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0
)

result = list(splitter.split_text(text))

ind = 8

if ind > len(result):
    print("Splitter out of range, ", len(result))
else:
    print(result[ind])