import itertools, random

random.seed(0)

subjects = ["the sensor module", "the wifi radio", "the battery controller", "the firmware loader",
            "the temperature probe", "the humidity sensor", "the motion detector", "the status LED",
            "the flash controller", "the power management unit", "the Bluetooth stack", "the sleep timer",
            "the audio codec", "the display driver", "the touch controller", "the accelerometer"]

actions = ["enters low power mode after", "reports an error when", "resets automatically if",
           "logs a warning to the console when", "requires recalibration after", "disables itself until",
           "sends a notification once", "throttles its sampling rate when", "reconnects to the network after",
           "flushes its buffer before", "wakes from deep sleep when", "skips the next reading if"]

conditions = ["the battery level drops below ten percent", "the ambient temperature exceeds safe limits",
              "a firmware update is pending", "the network connection is lost for more than a minute",
              "the device has been idle for five minutes", "a checksum mismatch is detected",
              "the configuration file is missing", "the watchdog timer expires",
              "the user presses the reset button", "the calibration data is out of date",
              "the sensor reading is out of range", "the storage partition is nearly full"]

topics = ["Wi-Fi station sleep mode", "deep sleep and light sleep configuration",
          "over-the-air firmware updates", "battery charging thresholds",
          "sensor calibration procedures", "error code reference tables",
          "Bluetooth Low Energy pairing", "flash partition layout",
          "power consumption tuning", "watchdog timer configuration",
          "audio codec sample rates", "display brightness control",
          "touch sensitivity calibration", "accelerometer axis mapping",
          "network reconnection backoff", "logging verbosity levels"]

templates = [
    "{subj} {act} {cond}.",
    "When {cond}, {subj} {act2}.",
    "Refer to the section on {topic} if {subj} {act} {cond}.",
    "To reduce power consumption, configure {subj} so it {act} {cond}.",
]

def act2(a):
    return a.replace("enters", "enter").replace("reports", "report").replace("resets", "reset") \
             .replace("logs", "log").replace("requires", "require").replace("disables", "disable") \
             .replace("sends", "send").replace("throttles", "throttle").replace("reconnects", "reconnect") \
             .replace("flushes", "flush").replace("wakes", "wake").replace("skips", "skip")

lines = set()
combos = list(itertools.product(subjects, actions, conditions))
random.shuffle(combos)
for subj, act, cond in combos:
    t = random.choice(templates)
    topic = random.choice(topics)
    s = t.format(subj=subj, act=act, act2=act2(act), cond=cond, topic=topic)
    s = s[0].upper() + s[1:]
    lines.add(s)
    if len(lines) >= 900:
        break

lines = sorted(lines)
random.shuffle(lines)
with open("corpus.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

# docs.txt: paragraph-style chunks, 2-3 sentences each, for the search demo corpus
chunks = []
for topic in topics:
    related = [l for l in lines if topic.lower() in l.lower()]
    random.shuffle(related)
    if len(related) >= 2:
        chunks.append(" ".join(related[:3]))
# pad with singleton chunks from remaining lines if needed
extra = [l for l in lines if not any(l in c for c in chunks)]
random.shuffle(extra)
chunks.extend(extra[: max(0, 40 - len(chunks))])
chunks = sorted(set(chunks))
random.shuffle(chunks)
with open("docs.txt", "w") as f:
    f.write("\n".join(chunks) + "\n")

print(f"{len(lines)} corpus lines, {len(chunks)} doc chunks")
