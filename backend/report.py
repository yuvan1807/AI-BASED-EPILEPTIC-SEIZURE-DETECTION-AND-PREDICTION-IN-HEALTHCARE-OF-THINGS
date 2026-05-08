from twilio.rest import Client

def send_whatsapp_report(prediction, confidence, seizure_conf, non_seizure_conf):
    
    account_sid = "ACfb1480d9e154ad5a16023f3b61e34954"
    auth_token = "170acecda10f2dcef161a3c4e95facdf"

    client = Client(account_sid, auth_token)

    message_body = f"""
EEG Seizure Detection Report

Result: {prediction}
Confidence: {confidence}

Seizure Probability: {seizure_conf}
Non-Seizure Probability: {non_seizure_conf}

Status: Analysis Completed
    """

    message = client.messages.create(
        body=message_body,
        from_='whatsapp:+14155238886',  # Twilio sandbox number
        to='whatsapp:+919047251565'     # Your WhatsApp number
    )

    print("WhatsApp message sent:", message.sid)