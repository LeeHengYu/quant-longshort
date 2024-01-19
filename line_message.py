import requests 
from os import environ

token = environ['LINE_CHANNEL_TOKEN']

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}

def LINEMessageCall(*text: str, notification = True, validate = False):
    """
    Send the messages to LINE Messaging API and broadcast.
    
    No more than 5 messages in one function call.
    
    Validate the messages if validation is set to True (default False).
    """
    if len(text) > 5: 
        raise ValueError("More than 5 messages called")
    
    validation_endpoint = "https://api.line.me/v2/bot/message/validate/broadcast"
    url = "https://api.line.me/v2/bot/message/broadcast"
        
    payload = {
        "messages": [],
        "notificationDisabled": not notification
    }
    
    for t in text:
        payload["messages"].append({
            "type": "text",
            "text": t,
        })
    
    if validate:
        isValid = requests.post(validation_endpoint, headers=headers, json=payload).status_code
        if isValid != 200:
            print("Error in broadcasting")
            return
    
    response = requests.post(url, headers=headers, json=payload)
    print("Message sent successfully.")

if __name__ == '__main__':
    LINEMessageCall("test", validate = True)