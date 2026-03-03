This script helps to configure the layer_1 environment. The environment consists of:
 * Mosquitto MQTT Broker
 * Influx DB
 * Telegraf

To prepare the environment, first you need to prepare JSON ```secrets.json``` and put it in this directory (```/layer_1/config```). JSON should have the following structure:

```json
{
    "Influx_Token" : "Your admin token to Influx Database",
    "Influx_Username" : "Your username for Influx Database",
    "Influx_Password" : "Password for your Influx account"
}
```

Later, to generate ```docker-compose.yml```, ```mosquitto.conf``` and ```telegraf.conf``` files, run the following script

``` bash
python configure.py
```

Having the files generated, just run the docker containers

```bash
docker compose up -d
```