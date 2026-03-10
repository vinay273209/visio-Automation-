import ssl
import paho.mqtt.client as mqtt
import cv2
import time
from cam import get_head_counts

BROKER = "z9262f1f.ala.eu-central-1.emqxsl.com"
PORT = 8883
USERNAME = "NIET_273209"
PASSWORD = "Yaniv@273209"

client = mqtt.Client(client_id="vision_ai_system")

client.username_pw_set(USERNAME, PASSWORD)
client.tls_set(cert_reqs=ssl.CERT_NONE)
client.tls_insecure_set(True)

client.connect(BROKER, PORT, 60)
client.loop_start()

fan_state = {
    "fan1": None,
    "fan2": None,
    "fan3": None,
    "fan4": None
}

last_check = time.time()
CHECK_INTERVAL = 5

last_student_time = time.time()
NO_STUDENT_TIMEOUT = 30


def control_fan(zone_count, fan_name, topic):

    desired_state = "ON" if zone_count > 1 else "OFF"

    if fan_state[fan_name] != desired_state:
        client.publish(topic, desired_state)
        fan_state[fan_name] = desired_state
        print(f"MQTT SENT -> {topic} : {desired_state}")


def turn_off_all_fans():

    for i in range(1,5):
        topic = f"home/fan{i}/set"
        if fan_state[f"fan{i}"] != "OFF":
            client.publish(topic, "OFF")
            fan_state[f"fan{i}"] = "OFF"
            print(f"MQTT SENT -> {topic} : OFF")


while True:

    z1, z2, z3, z4 = get_head_counts()

    print("Zone1:", z1, "Zone2:", z2, "Zone3:", z3, "Zone4:", z4)

    total_students = z1 + z2 + z3 + z4

    # update last detection time
    if total_students > 0:
        last_student_time = time.time()

    # fan control every 5 seconds
    if time.time() - last_check > CHECK_INTERVAL:

        last_check = time.time()

        control_fan(z1, "fan1", "home/fan1/set")
        control_fan(z2, "fan2", "home/fan2/set")
        control_fan(z3, "fan3", "home/fan3/set")
        control_fan(z4, "fan4", "home/fan4/set")

        print("------ FAN STATUS UPDATED ------")

    # no student for 30 seconds → turn off all fans
    if time.time() - last_student_time > NO_STUDENT_TIMEOUT:

        print("NO STUDENTS FOR 30s → TURNING OFF ALL FANS")

        turn_off_all_fans()

        last_student_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()