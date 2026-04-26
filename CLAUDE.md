# PiRobot — Autonomous Pet Robot

## Goal

Turn the Freenove 4WD Smart Car (Raspberry Pi 4B/5) into an autonomous pet robot that roams its
environment with believable personality: curiosity, playfulness, fear responses, boredom, and
sleep/wake cycles. It expresses its emotional state through LED colors and buzzer chirps.

---

## Architecture

```
autonomous3.py       Main loop — entry point for pet mode
pet_brain_local.py   Local rule-based pet brain (no server needed)
ollama_brain.py      Ollama LLM brain (subclass of LocalPetBrain, uses llama3.2:1b)
car.py               Hardware abstraction: motors, servos, sensors, dead-reckoning, occupancy memory
led.py               NeoPixel LED strip (8 pixels)
buzzer.py            Piezo buzzer
camera.py            Picamera2 (optional, off by default)
pet_server_bridge.py HTTP bridge to remote AI server (optional)
parameter.py         Reads params.json — hardware version config
```

### Three brain modes

| Priority | Mode | When | File |
|----------|------|------|------|
| 1 | **Server brain** | `PET_SERVER_URL` is set | `autonomous3.ServerPetBrain` |
| 2 | **Ollama brain** | `OLLAMA_URL` is set | `ollama_brain.OllamaBrain` |
| 3 | **Local brain** | neither env var set | `pet_brain_local.LocalPetBrain` |

All brains expose the same interface:
```python
brain.tick(car, led, buzzer, now: float, dt: float)
brain.stuck_recent: bool       # written by main loop
```

**Local brain** is fully self-contained — no network calls.  
**Ollama brain** extends LocalPetBrain: all safety reflexes and emotion logic run as-is; every
~2.5 s it asks `llama3.2:1b` on the local Ollama server for a structured action decision and
applies it when fresh. Falls back silently to the parent rule-based pick when the LLM is busy
or after 8 consecutive failures. Optionally uses `moondream:1.8b` for async scene captions.  
**Server brain** sends a sensor snapshot to `POST {PET_SERVER_URL}/pet/step` and applies the
returned action plan. Falls back to LocalPetBrain after 5 consecutive server failures.

---

## Running the robot

```bash
# Headless pet mode — local rule-based brain
python autonomous3.py

# Ollama brain (llama3.2:1b on the Pi itself)
OLLAMA_URL=http://localhost:11434 python autonomous3.py

# Ollama on another machine on the network
OLLAMA_URL=http://192.168.x.x:11434 python autonomous3.py

# Remote AI server brain
PET_SERVER_URL=http://192.168.x.x:8000 python autonomous3.py

# Full GUI server (manual / line / light / ultrasonic modes)
python main.py
python main.py --terminal   # headless
```

---

## Environment variables (`.env` or export)

| Variable | Default | Description |
|----------|---------|-------------|
| `PET_SERVER_URL` | *(empty)* | Remote AI server URL. If blank → local/Ollama brain |
| `OLLAMA_URL` | *(empty)* | Ollama base URL e.g. `http://localhost:11434`. Used when `PET_SERVER_URL` is not set |
| `OLLAMA_LLM_MODEL` | `llama3.2:1b` | Ollama model for action decisions |
| `OLLAMA_VIS_MODEL` | `moondream:1.8b` | Ollama model for scene captions (optional) |
| `OLLAMA_TIMEOUT_S` | `8` | Ollama request timeout in seconds |
| `OLLAMA_LLM_DT_S` | `2.5` | How often to request a new LLM decision |
| `PI_API_KEY` | *(empty)* | API key sent as `X-API-Key` header (server brain) |
| `PET_ROBOT_ID` | `sparky` | Robot identity string (≤32 chars) |
| `PET_ANON` | `1` | Anonymise robot ID in server payloads |
| `PET_PRIVACY_LEVEL` | `normal` | `normal` / `minimal` / `offload` |
| `PET_STEP_TIMEOUT_S` | `6` | Server request timeout in seconds |
| `PET_USE_CAMERA` | `0` | Enable camera/target-tracker (`1` to enable) |
| `LOW_BATTERY_THRESHOLD` | `5.0` | Volts below which the robot sleeps |
| `BATTERY_FULL_V` | `8.4` | Voltage corresponding to 100% battery |
| `BATTERY_EMPTY_V` | `5.5` | Voltage corresponding to 0% battery |

---

## Pet behavior design

### Emotional states (local brain)

| Emotion | LED color | Chirp | Behavior |
|---------|-----------|-------|----------|
| `CURIOUS` | Cyan | 1× short | Wander at normal speed; head-scan when stopped |
| `HAPPY` | Warm yellow | 3× short | Faster wander; upbeat movement |
| `PLAYFUL` | Purple | 2× medium | Executes a play sequence (zoomies / wiggle / spin), then transitions to HAPPY |
| `BORED` | Dim amber | — | Mostly stopped; slow head sweeps; rare short forward nudge |
| `TIRED` | Dim blue | 1× long | Slow wander; frequent pauses |
| `SCARED` | Red | 4× rapid | Spin or reverse away from repeated close obstacles |
| `SLEEP` | Very dim blue | — | Stopped; energy recharges; wakes at 55% energy |

### State transitions

```
Start → CURIOUS
  high energy + time since play > 30s → PLAYFUL → HAPPY
  obstacle < 55 cm → CURIOUS (heightened)
  obstacle < 28 cm ×3 in 6s → SCARED (3.5s)
  energy < 38% → TIRED
  energy < 12% → SLEEP → (recharge to 55%) → HAPPY
  idle > 20s → BORED
```

### Energy dynamics

| State | Rate |
|-------|------|
| SLEEP | +0.022 / s |
| TIRED | +0.003 / s |
| BORED / CURIOUS | −0.003–0.005 / s |
| HAPPY | −0.007 / s |
| PLAYFUL | −0.015 / s |
| SCARED | −0.010 / s |

---

## Safety reflexes (always active, both brains)

These run in the main loop before the brain tick and override any brain action:

| Trigger | Response |
|---------|----------|
| Battery < 5 V | Stop motors, enter SLEEP |
| IR sensors all-zero while moving forward | Reverse 220 ms + spin (cliff / edge detection) — disabled by default; set `PET_CLIFF_DETECT=1` |
| Ultrasonic < 30 cm | Hard stop + `scan_and_avoid_with_memory()` (both main loop & LocalPetBrain) |
| Ultrasonic < 65 cm | `scan_and_avoid_with_memory()` (both main loop & LocalPetBrain) |
| Ultrasonic 65–90 cm | Graduated speed reduction (25–100 % of target speed) |
| Stuck (distance unchanged > 1.6 s while forward) | `escape_stuck_with_memory()` |

---

## Hardware config (`params.json`)

```json
{
  "Connect_Version": 2,     // 1 = GPIO LEDs (Pi 4B), 2 = SPI LEDs
  "Pcb_Version": 1,         // 1 = 3.3 V ADC,  2 = 5.2 V ADC
  "Pi_Version": 1,          // 1 = Pi 4B,       2 = Pi 5
  "Servo_Trim": {"0": 0, "1": -8}
}
```

GPIO pins: ultrasonic trigger=27, echo=22 · IR=14/15/23 · buzzer=17

---

## Key implementation notes

- **No wheel encoders** — dead-reckoning only (25 cm/s @ PWM 600; 220 °/s @ PWM 1500).
  Accuracy degrades over ~5 minutes; the occupancy grid compensates by scoring unexplored cells.
- **Head = sonar** — the ultrasonic is mounted on the pan servo.  
  Always call `car.park_head_for_drive()` before moving forward so `get_forward_distance()`
  reads straight ahead. Never rotate the head while driving forward.
- **Speed scale** — `Car.set_motors()` applies a global 0.65× scale and enforces a 260 PWM
  deadband.  Pass raw intent values (e.g. 1100); the scaling happens internally.
- **Loop rate** — `LOOP_DT = 0.05 s` (~20 Hz). Sensor update intervals are staggered to avoid
  I²C/GPIO contention: battery 600 ms, sonar 180 ms, IR 120 ms, light 500 ms, pose 350 ms.
- **`led.ledIndex(0xFF, R, G, B)`** sets all 8 pixels simultaneously.
- Adding a new emotion: add it to `LED_COLORS`, `CHIRP_PATTERNS`, `_update_emotion()`, and
  `_pick_action()` in `pet_brain_local.py`.
