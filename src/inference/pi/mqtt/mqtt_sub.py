import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
TOPIC = "edgeai/face_classifier"

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    print("Topic:", msg.topic)
    print("Payload:", msg.payload.decode("utf-8"))
    print("-" * 50)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)

print("Waiting for data...")
client.loop_forever()