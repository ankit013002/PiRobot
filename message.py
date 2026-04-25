class MessageParser:
    """
    Parse the robot protocol format: CMD_NAME#param1#param2...

    After parse(), use:
        .command   — command string (e.g. "CMD_MOTOR")
        .params    — list of numeric parameters (float → rounded to int)
    """

    def __init__(self):
        self.command: str        = ""
        self.params:  list[int]  = []

    def parse(self, msg: str) -> bool:
        self.command = ""
        self.params  = []
        try:
            parts = msg.strip().split('#')
            self.command = parts[0]
            for token in parts[1:]:
                token = token.strip()
                if not token:
                    continue
                try:
                    self.params.append(round(float(token)))
                except ValueError:
                    # Legacy voice-word fallback (one/two/three/four)
                    self.params.append({'one': 1, 'two': 2, 'three': 3, 'four': 4}.get(token.lower(), 0))
            return True
        except Exception as e:
            print(f"[MessageParser] error: {e}  raw={msg!r}")
            return False

    # Backward-compat aliases used by main.py
    @property
    def command_string(self) -> str:
        return self.command

    @property
    def int_parameter(self) -> list[int]:
        return self.params

    def clear_parameters(self) -> None:
        self.command = ""
        self.params  = []


# Keep old name as alias so existing code that imports Message_Parse still works
Message_Parse = MessageParser


if __name__ == '__main__':
    p = MessageParser()
    for raw in ["CMD_LED#0#255#0#0", "CMD_MOTOR#1200#1200#1200#1200", "CMD_MODE#3"]:
        p.parse(raw)
        print(f"{p.command}  {p.params}")
