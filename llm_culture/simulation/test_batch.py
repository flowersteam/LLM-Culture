from server_answer import get_answer
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


prompts = ['Tell me a joke', 'Tell me a fact','What is the weather today?', 'How old are you?', 'What is your favorite color?', 'What is your name?', 'What is the size of the sun?', 'What is the size of the moon?' ]


access_url ='https://remedy-manage-italian-impacts.trycloudflare.com'


url = access_url + "/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}
datas = []


import time

start_time = time.time()


for prompt in prompts:
    history = []

    #prompt = '<|im_start|>user' + prompt + '<|im_end|> <|im_start|>assistant'

    history.append({"role": "user", "content": prompt})
    data = {
        "mode": "chat",
        "role": "assistant",
        "messages": history
    }

    datas.append(data)

for d in datas:
    while True:
        response = requests.post(url, headers=headers, json=d, verify=False)
        try:
            assistant_message = response.json()['choices'][0]['message']['content'].replace('</s>', '')
            print(assistant_message)
            break
        except:
            print('No answer from server, trying again...')

end_time = time.time() 
execution_time = end_time - start_time 
print("Execution time:", execution_time)


start_time = time.time()

for prompt in prompts:
    a = get_answer(access_url, prompt)
    print(a)

end_time = time.time() 
execution_time = end_time - start_time 
print("Execution time:", execution_time)




#history.append({"role": "assistant", "content": assistant_message})
#stories.append("Story" +str(i)+": " + assistant_message)
#print("Answer: " +assistant_message)
