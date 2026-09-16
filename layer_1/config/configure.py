import subprocess
import json
from pathlib import Path

def run_configure(secrets, context):

  DOCKER_COMPOSE_FILE = f"""
  version: '3.8'

  services:

    influxdb:
      image: influxdb:2.7
      container_name: influxdb
      ports:
        - "8086:8086"
      volumes:
        - influxdb_data:/var/lib/influxdb2
      environment:
        DOCKER_INFLUXDB_INIT_MODE: setup
        DOCKER_INFLUXDB_INIT_USERNAME: {secrets["Influx_Username"]}
        DOCKER_INFLUXDB_INIT_PASSWORD: {secrets["Influx_Password"]}
        DOCKER_INFLUXDB_INIT_ORG: FeederPW
        DOCKER_INFLUXDB_INIT_BUCKET: Feeder
        DOCKER_INFLUXDB_INIT_ADMIN_TOKEN: {secrets["Influx_Token"]}
      restart: unless-stopped

    mosquitto:
      image: eclipse-mosquitto
      container_name: mosquitto
      ports:
        - "1883:1883"
      volumes:
        - ./mosquitto.conf:/mosquitto/config/mosquitto.conf
      restart: unless-stopped

    telegraf:
      image: telegraf:latest
      container_name: telegraf
      depends_on:
        - influxdb
        - mosquitto
      volumes:
        - ./telegraf.conf:/etc/telegraf/telegraf.conf:ro
      restart: unless-stopped

  volumes:
    influxdb_data:
  """

  MOSQUITTO_CONF = f"""
  allow_anonymous true
  listener 1883
  """

  TELEGRAF_CONF = f"""
  [agent]
    interval = "10s"
    flush_interval = "10s"

  [[outputs.influxdb_v2]]
    urls = ["http://influxdb:8086"]
    token = "{secrets["Influx_Token"]}"
    organization = "FeederPW"
    bucket = "Feeder"

  [[inputs.mqtt_consumer]]
    servers = ["tcp://mosquitto:1883"]
    topics = [
      "Feeder/IoT/#"
    ]

    topic_exclude = [
      "Feeder/IoT/bridge/#"
    ]

    data_format = "json"

  [[processors.converter]]
    [processors.converter.fields]
      integer = ["trigger_indicator"]
  """

  with open(context / "mosquitto.conf", "w") as mosquitto_file:
      mosquitto_file.write(MOSQUITTO_CONF)

  with open(context / "docker-compose.yml", "w") as docker_file:
      docker_file.write(DOCKER_COMPOSE_FILE)

  with open(context / "telegraf.conf", "w") as telegraf_file:
      telegraf_file.write(TELEGRAF_CONF)

if __name__ == "__main__":
  print("Running configuration")
  
  CONTEXT = Path(__file__).parent

  with open(CONTEXT / "secrets.json", "r") as f:
      secrets = json.load(f)

  run_configure(secrets, CONTEXT)
  
  print("Done")