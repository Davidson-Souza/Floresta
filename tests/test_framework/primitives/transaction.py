BECH32M_CONST = 0x2BC830A3


class Script:
    def __init__(self, script: bytes):
        self.script = script

    def __repr__(self):
        return f"Script(script={self.script.hex()})"

    def bytes(self) -> bytes:
        # bech32 encodes the script into a Bitcoin address.
        return self.script


class Amount:
    def __init__(self, value: int):
        self.value = value

    def __repr__(self):
        return f"Amount(value={self.value})"


class OutPoint:
    def __init__(self, txid: str, index: int):
        self.txid = txid
        self.index = index


class TxOutput:
    def __init__(self, script: Script, amount: Amount):
        self.script = script
        self.amount = amount


class TxInput:
    def __init__(
        self, outpoint: OutPoint, script_sig: Script, sequence: int = 0xFFFFFFFF
    ):
        self.outpoint = outpoint
        self.script = script_sig
        self.sequence = sequence

    def __repr__(self):
        return f"TxInput(outpoint={self.outpoint}, script={self.script}, sequence={self.sequence})"


class Transaction:
    def __init__(
        self,
        version: int,
        inputs: list[TxInput],
        outputs: list[TxOutput],
        locktime: int = 0,
    ):
        self.version = version
        self.inputs = inputs
        self.outputs = outputs
        self.locktime = locktime

    def __repr__(self):
        return (
            f"Transaction(version={self.version}, "
            f"inputs={self.inputs}, "
            f"outputs={self.outputs}, "
            f"locktime={self.locktime})"
        )

    def segwit_sighash(self, input_index: int, hash_type: int) -> bytes:
        # This is a placeholder for the actual SegWit sighash calculation.
        # In a real implementation, this would compute the correct hash based on the transaction data.
        return b"\x00" * 32

    def add_input(self, tx_input: TxInput):
        self.inputs.append(tx_input)

    def add_output(self, tx_output: TxOutput):
        self.outputs.append(tx_output)

    def get_input(self, index: int) -> TxInput:
        if index < 0 or index >= len(self.inputs):
            raise IndexError("Input index out of range")
        return self.inputs[index]

    def get_output(self, index: int) -> TxOutput:
        if index < 0 or index >= len(self.outputs):
            raise IndexError("Output index out of range")
        return self.outputs[index]

    def compute_fees(self, prevouts: list[TxOutput]) -> Amount:
        total_input_value = sum(output.amount.value for output in prevouts)
        total_output_value = sum(output.amount.value for output in self.outputs)
        fee_value = total_input_value - total_output_value
        return Amount(fee_value)

    def is_segwit(self) -> bool:
        # This is a placeholder for the actual check to determine if the transaction is SegWit.
        # In a real implementation, this would check if any inputs have witness data.
        return False

    def serialize(self) -> bytes:
        # This is a placeholder for the actual serialization logic.
        # In a real implementation, this would convert the transaction to its byte representation.
        return b""

    def deserialize(self, data: bytes):
        # This is a placeholder for the actual deserialization logic.
        # In a real implementation, this would parse the byte data into a Transaction object.
        pass

    def txid(self) -> str:
        # This is a placeholder for the actual transaction ID calculation.
        # In a real implementation, this would compute the transaction ID based on the serialized data.
        return "dummy_txid"

    def wtxid(self) -> str:
        # This is a placeholder for the actual witness transaction ID calculation.
        # In a real implementation, this would compute the witness transaction ID based on the serialized data.
        return "dummy_wtxid"

    def __eq__(self, other):
        if not isinstance(other, Transaction):
            return False

        return self.wtxid() == other.wtxid()
