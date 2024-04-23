# This file is used to get the answer from the server. 
#It is called by the agent.py file.
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_answer(access_url, prompt, debug=False):
    url = access_url + "/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

    history = []

    #prompt = '<|im_start|>user' + prompt + '<|im_end|> <|im_start|>assistant'
    
    history.append({"role": "user", "content": prompt})
    data = {
        "mode": "chat",
        "role": "assistant",
        "messages": history
    }

    while True:
        response = requests.post(url, headers=headers, json=data, verify=False)
        try:
            assistant_message = response.json()['choices'][0]['message']['content'].replace('</s>', '')
            break
        except:
            print('No answer from server, trying again...')

    #history.append({"role": "assistant", "content": assistant_message})
    #stories.append("Story" +str(i)+": " + assistant_message)
    #print("Answer: " +assistant_message)

    return assistant_message



  
# if __name__ == "__main__":
#     print(get_answer('https://sides-create-born-institute.trycloudflare.com', "Tell me a joke"))