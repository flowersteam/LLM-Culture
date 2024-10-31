import requests

def get_answer(access_url, prompt, debug=False):
    """Get the answer from the server

    :param access_url: url to access the server
    :param prompt: prompt to send to the server
    :param debug: wether to debug or not, defaults to False
    :return: the answer from the server
    """
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

    return assistant_message