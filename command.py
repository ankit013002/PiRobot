# Protocol command strings sent between client and server.
# Format: CMD_NAME#param1#param2...\n

CMD_MOTOR      = "CMD_MOTOR"       # FL BL FR BR  (±4095 PWM)
CMD_M_MOTOR    = "CMD_M_MOTOR"     # mecanum: angle speed angle speed
CMD_CAR_ROTATE = "CMD_CAR_ROTATE"  # x y angle time
CMD_LED        = "CMD_LED"         # index R G B
CMD_LED_MOD    = "CMD_LED_MOD"     # mode (0–5)
CMD_SERVO      = "CMD_SERVO"       # channel angle
CMD_BUZZER     = "CMD_BUZZER"      # 0|1
CMD_SONIC      = "CMD_SONIC"       # request distance
CMD_LIGHT      = "CMD_LIGHT"       # request light levels
CMD_POWER      = "CMD_POWER"       # request battery voltage
CMD_MODE       = "CMD_MODE"        # mode index (0=manual 1=light 2=IR 3=sonic)
CMD_LINE       = "CMD_LINE"        # request IR line sensor values
