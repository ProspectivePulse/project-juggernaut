import socket, json, time
from megapi_python3 import MegaPi
# from threading import Event
# import rc_callbacks
# from rc_callbacks import ultrasonic_cb, current_distance, stop_ultrasonic_evt

HOST = "0.0.0.0"
PORT = 5000
SERIAL_DEV = "/dev/ttyUSB0"
ULTRASONIC_PORT = 8
MOTOR_SPEED = 255
STEP_SEC = 0.40

ACTIONS = ["forward", "backward", "left", "right", "stop"]

# --- Ultrasonic callback --- #
current_distance = None
cb_hits = 0
last_update_ts = 0.0
last_arm_ts = 0.0
ARM_COOLDOWN = 0.8

def ultrasonic_cb(value):	
	global current_distance, cb_hits, last_update_ts
	try:
		current_distance = value
		cb_hits += 1
		last_update_ts = time.time()
		# print("[CB]", current_distance)
	except Exception:
		pass

def reopen_serial():
	global bot
	try:
		try: bot.close()
		except: pass
		time.sleep(0.2)
		bot = MegaPi(); bot.start(SERIAL_DEV)
		print(f"Reopened serial on {SERIAL_DEV}")
		return True
	except Exception as e:
		print("Reopen serial failed:", e)
		return False


def arm_ultrasonic(bot, port):

	global last_arm_ts

	if time.time() - last_arm_ts < ARM_COOLDOWN:
		return

	try:
		bot.ultrasonicSensorRead(port, ultrasonic_cb)
		print(f" Ultrasonic armed on port {port}")
	except Exception as e:
		print(" Ultrasonic start failed:", e)


def take_action(bot, idx: int):

	a = ACTIONS[idx]
	if a == "forward":
		bot.encoderMotorRun(1, MOTOR_SPEED); bot.encoderMotorRun(2, -MOTOR_SPEED)
	elif a == "backward":
		bot.encoderMotorRun(1, -MOTOR_SPEED); bot.encoderMotorRun(2, MOTOR_SPEED)
	elif a == "left":
		bot.encoderMotorRun(1, MOTOR_SPEED); bot.encoderMotorRun(2, MOTOR_SPEED)
	elif a == "right":
		bot.encoderMotorRun(1, -MOTOR_SPEED); bot.encoderMotorRun(2, -MOTOR_SPEED)
	else:
		bot.encoderMotorRun(1, 0); bot.encoderMotorRun(2, 0)
	time.sleep(STEP_SEC)
	bot.encoderMotorRun(1, 0); bot.encoderMotorRun(2, 0)

def get_obs():
	d = current_distance

	if d is None: return 0.5
	d = max(20.0, min(400.0, float(d)))
	return d / 400.0

def get_reward(raw_cm, act_idx):
	if raw_cm is None:
		return 0.0

	if act_idx == 0:
		if raw_cm > 100.0:
			return min(5.0, raw_cm/20.0)
		else:
			return -10.0/max(1.0, raw_cm)

	if act_idx == 4 and raw_cm > 100.0:
		return -1.0

	
	if raw_cm <= 100.0 and act_idx == 1:
		return 2.0
	elif raw_cm <= 100.0 and act_idx in [3, 2]:
		return 1.0
	elif raw_cm <= 100.0 and act_idx == 4:
		return 0.0
	
	return -0.01	

def main():
	# --- hardware init --- #
	print("Init MegaPi")
	bot = MegaPi()
	try:
		bot.start(SERIAL_DEV)
		print(f"MegaPi started on {SERIAL_DEV}")
	except Exception as e:
		print(f"MegaPi start failed on {SERIAL_DEV}:", e)
		bot = None	


	try:
		bot.ultrasonicSensorRead(ULTRASONIC_PORT, ultrasonic_cb)
		print(f" Ultrasonic armed on port {ULTRASONIC_PORT}")
	except Exception as e:
		print(f" Ultrasonic start failed:", e)

	t0 = time.time()
	while current_distance is None and time.time() - t0 < 2.0:
		time.sleep(0.05)
	if current_distance is None:
		print("WARNING: No ultrasonic reading within 2s; continuing best-effort.")

	# --- socket server --- #
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.bind((HOST, PORT)); s.listen(1)
	print(f"Pi RL listening on {HOST}:{PORT}")
	conn, addr = s.accept()
	print(f"Connection from {addr}")

	try:
		while True:
			data = conn.recv(1024)
			if not data:			
				time.sleep(0.01)
				continue
			try:
				msg = json.loads(data.decode())
			except json.JSONDecodeError:
				continue # partial packet; wait for more
			

			act_idx = int(msg.get("action", -1))

			
			if (time.time() - last_update_ts) > 2.0:
				print(" ultrasonic stale; rearming...")
				try:
					arm_ultrasonic(bot, ULTRASONIC_PORT)
				except Exception as e:
					print(" rearm error:", e)
					if reopen_serial():
						arm_ultrasonic(bot, ULTRASONIC_PORT)

			if not hasattr(main, "_last_hits"):
				main._last_hits = -1
			if cb_hits != main._last_hits:
				#print(f"[CB] hits={cb_hits} last raw_cm={current_distance}")
				main._last_hits = cb_hits	

			if act_idx == -1:
				raw = current_distance
				obs = get_obs()
				reply = {"obs":obs, "reward":0.0, "done":False, "raw_cm":raw}
				# print(f"[SERVER] act={act_idx} raw_cm={raw}")
				conn.send(json.dumps(reply).encode())
				continue

			if 0 <= act_idx < len(ACTIONS):
				take_action(bot, act_idx)

			time.sleep(0.05)
			raw = current_distance
			if raw is None or raw <= 110 and act_idx == 0:
				print("Safety override: stop instead of forward")
				act_idx = 4
			obs = get_obs()
			reply = {"obs":obs, "reward":get_reward(raw, act_idx), "done":False, "raw_cm":raw}
			print(f"[SERVER], {reply}, action: {ACTIONS[act_idx]}")
			conn.send(json.dumps(reply).encode())

	except Exception as e:
		print("Loop Error:", e)
		


	finally:
		try:			
			bot.encoderMotorRun(1, 0); bot.encoderMotorRun(2, 0)
		except Exception:
			pass
		try:
			bot.close()
		except Exception:
			pass
		try:
			conn.close(); s.close()
		except Exception:
			pass
		print("Server closed")

if __name__ == "__main__":
	main()
