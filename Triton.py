from typing import Any, Iterable, Tuple, Union
from qcodes import VisaInstrument
from qcodes.instrument.parameter import Parameter, ParameterWithSetpoints
import qcodes.utils.validators as vals

def configure_socket(address: str) -> str:
    oxford_ins_port = 33576
    return "TCPIP0::" + address + f"::{oxford_ins_port}::SOCKET"

class Triton200(VisaInstrument):
    """
    This is a qcodes implementation of the driver for an 
    Oxford Instruments Triton 200 cryostat
    """

    def __init__(self, name: str, address: str, **kwargs: Any):
        super().__init__(name, address, **kwargs)
        self.host = address

        self.add_parameter(
            "turbo1",
            label='Turbo 1',
            get_cmd="READ:DEV:TURB1:PUMP:SIG:STATE",
            #set_cmd="SET:DEV:TURB1:PUMP:SIG:STATE:ON"
        )

class TurboPump(Parameter):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.base_str = ":DEV:" + name.replace('o', '').upper() + ":PUMP:SIG"
        self.read_str = self.

    def get_raw(self) -> str:
        # self.root_instrument.write("READ" + self.base_str + "STATE:SPD")
        # return self.root_instrument.visa_handle.read_raw()
        return self.root_instrument.ask("READ" + self.base_str + "STATUS") #oneliner to replace the two above
    
    def set_raw(self, state: str) -> str:
        self.root_instrument.write("SET" + self.base_str + "STATE:" + state.upper())

    def speed(self) -> str:
        cmd_string = "READ" + self.base_str + "SPD"
        result_string = self.root_instrument.ask(cmd_string)
        return result_string[len(cmd_string) + 1:]
    
    def state(self) ->

    
def get_values(module: object, base_str: str, *commands) -> str:
    cmd_string = ''
    for command in commands:
        cmd_string += str(command + ':')
    return module.root_instrument.ask("READ" + base_str + cmd_string[:-1])