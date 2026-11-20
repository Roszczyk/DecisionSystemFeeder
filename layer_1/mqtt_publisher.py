import paho.mqtt.client as mqtt

class MQTT_Publisher:
    def __init__(self, mqtt_broker_host : str, mqtt_broker_port : int, username, password):
        self.client = mqtt.Client()
        self.client.username_pw_set(username, password)
        self.client.connect(mqtt_broker_host, mqtt_broker_port)

    def publish(self, topic, content):
        self.client.publish(topic, content)