from test_framework.primitives.segwit_address import Encoding, encode, decode
from test_framework.primitives.transaction import Script


class Bech32Address:
    def __init__(
        self, script: Script, encoding: Encoding = Encoding.BECH32, hrp: str = "bcrt"
    ):
        self.script = script
        self.encoding = encoding
        self.hrp = hrp

    def to_string(self) -> str:
        script = self.script.bytes()
        witver = script[0]
        witprog = script[2:]  # skip the PUSH_BYTES OP
        return encode(self.hrp, witver, witprog)

    @staticmethod
    def from_string(addr: str, hrp="bcrt") -> "Address":
        hrp, data = decode(hrp, addr)
        if hrp is None or data is None:
            raise ValueError("Invalid address format")

        return Bech32Address(Script(data))

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()
