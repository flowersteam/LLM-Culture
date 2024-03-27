import ollama
from ollama import Client




#ollama.pull('mistral')
#print(ollama.generate(model='mistral', prompt='Why is the sky blue?', stream = True))

stream = ollama.chat(
    model='mistral',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)


