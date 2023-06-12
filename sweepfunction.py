from triton import OxfordTriton
from typing import Tuple, Optional, NoReturn
import numpy as np
from time import sleep

from qcodes.dataset import Measurement
from qcodes.parameters import Parameter, ParameterBase
from qcodes.instrument import Instrument


class Temperature(Parameter):
    def __init__(self, name: str, fridge: Instrument, t_sample_ch: int, *args, **kwargs):
        self.temperature_measurement = eval(f'fridge.T{t_sample_ch}')
        super().__init__(name=name, unit='K', label='Temperature', *args, **kwargs)

    def get_raw(self):
        return self.temperature_measurement()


def move_temp_setpoint(fridge: OxfordTriton, set_point: float) -> None:
    fridge.pid_setpoint(set_point)


def toggle_turbo(
        fridge: OxfordTriton, 
        future_setpoint: float, 
        current_temp: float,
        turbo_state: str,
        t_sample_ch: int,
    ) -> str:
    
    state = {1: 'on', -1: 'off'}
    critical_temp = .8
    critical_speed = 100
    best_state = state[np.sign(critical_temp - future_setpoint)]

    if best_state != turbo_state:

        if best_state == 'on': 
            while current_temp > critical_temp:
                current_temp = eval(f'fridge.T{t_sample_ch}()')
                sleep(1)
            fridge.turb1_state(best_state)

        else:
            fridge.turb1_state(best_state)
            while fridge.turb1_speed() > critical_speed:
                sleep(1)
                
        print('Turbo 1 has been switched ' + best_state)
        return best_state
    
    else:
        return turbo_state
        

def _set_heater_range(
        fridge: OxfordTriton, 
        future_setpoint: float, 
        current_heater_range: float
    ) -> float:

    for temp, curr in zip(fridge._heater_range_temp, fridge._heater_range_curr):
        if future_setpoint < temp:
            if current_heater_range != curr:
                fridge.pid_range(curr)
                print(f"Heater range changed to {curr} mA.")
                return curr
            else:
                return current_heater_range

def _get_sample_temp(fridge: OxfordTriton, t_sample_ch: int) -> float:
    return eval(f'fridge.T{t_sample_ch}()')


def _set_temp_setpoint(
        fridge: OxfordTriton, 
        set_point: float, 
        magnet_active: bool, 
        t_magnet_ch: int,
        turbo_state: str,
        heater_range: float,
    ) -> None:

    if magnet_active == True:
        magnet_safe(fridge, t_magnet_ch)

    current_temp = sample_temp()

    move_temp_setpoint(fridge, set_point, current_temp)
    
    turbo_state = toggle_turbo(fridge, set_point, turbo_state, current_temp)
    heater_range = _set_heater_range(fridge, set_point, heater_range)

    return turbo_state, heater_range


def magnet_safe(
        fridge: OxfordTriton, 
        t_magnet_ch: int
    ) -> Optional[NoReturn]:

    magnet_temp = eval(f'fridge.T{t_magnet_ch}()')
    if magnet_temp > 4.6:
        move_temp_setpoint(fridge, 0)
        raise ValueError('Magnet temperature is above 4.6K. \
                         Cool down or set magnet_active=False to deactivate magnet.')
    

def set_pid_controller(fridge: OxfordTriton, pid_values: Tuple[float, float, float]) -> None:
    P, I, D = pid_values
    fridge.pid_p(P)
    fridge.pid_i(I)
    fridge.pid_d(D)
    print(f'P = {P}, I = {I}, D = {D}')


def set_ramp_rate(fridge: OxfordTriton, rate: float) -> None:
    fridge.pid_rate(rate)
    print(f'Ramp rate set to {rate}')


def init_sweep_state(
                fridge: OxfordTriton,
                t_sample_ch: int,
                t_magnet_ch: int,
                magnet_active: bool,
                #ramp_rate: float,
                pid: Tuple[float, float, float],
    ) -> Tuple[str, str]:

    if magnet_active:
        magnet_safe(fridge, t_magnet_ch)
    print('Magnet is safe')

    keep_on_channels = [t_sample_ch, t_magnet_ch]
    keep_on_channels = ['T%s' % ch for ch in keep_on_channels]
    for ch in fridge.chan_temps.difference(keep_on_channels):
        eval(f'fridge.' + ch + '_enable("OFF")')

    set_pid_controller(fridge, pid)
    
    #set_ramp_rate(fridge, ramp_rate)

    return fridge.turb1_state(), fridge.pid_range()


def _close(temperature: float, setpoint: float, tolerance: float=1e-4):
    if abs(temperature - setpoint) < tolerance:
        return True
    else:
        return False


def T1dMeasurement(
            fridge: OxfordTriton,
            start: float,
            end: float,
            points: int,
            wait_time: float,
            #ramp_rate: float,
            *param_meas,
            pid_values: Tuple[float, float, float]=(10., 20., 0.),
            t_sample_ch: int=8,
            t_magnet_ch: int=13,
            magnet_active: bool=True,
            write_period: float=5.0,
):
    assert type(t_sample_ch) == int
    assert type(t_magnet_ch) == int
    assert type(points) == int
    
    sample_temp = Temperature('sample_temp', fridge, t_sample_ch)

    turbo_state, heater_range = init_sweep_state(
                                            fridge, 
                                            t_sample_ch, 
                                            t_magnet_ch, 
                                            magnet_active, 
                                            #ramp_rate,
                                            pid_values,
    )
    
    meas = Measurement()
    meas.write_period = write_period
    meas.register_parameter(sample_temp)
    params = []
    for param in param_meas:
        if isinstance(param, ParameterBase):
            params.append(param)
            meas.register_parameter(param, setpoints=(sample_temp,))

    with meas.run() as datasaver:

        setpoint_list = np.linspace(start, end, points)
        for setpoint in setpoint_list:
            turbo_state, heater_range = _set_temp_setpoint(
                                                        fridge, 
                                                        setpoint, 
                                                        magnet_active, 
                                                        t_magnet_ch,
                                                        turbo_state,
                                                        heater_range
            )

            while not _close(sample_temp(), setpoint):
                sleep(0.5)

            current_temperature = sample_temp()
            params_get = [(param, param.get()) for param in params]
            datasaver.add_result((sample_temp, current_temperature), *params_get)
            sleep(wait_time)

        return datasaver.dataset
    

def T2dMeasurement(
            fridge: OxfordTriton,
            start_temp: float,
            end_temp: float,
            points_temp: int,
            wait_time_temp: float,
            #ramp_rate: float,
            parameter_fast: ParameterBase,
            start_fast: float,
            end_fast: float,
            points_fast: int,
            wait_time_fast: float,
            *param_meas,
            pid_values: Tuple[float, float, float] = (10., 20., 0.),
            t_sample_ch: int=8,
            t_magnet_ch: int=13,
            magnet_active: bool=True,
            write_period: float=5.0,
):
    assert type(t_sample_ch) == int
    assert type(t_magnet_ch) == int
    assert type(points_temp) == int
    assert type(points_fast) == int

    sample_temp = Temperature('sample_temp', fridge, t_sample_ch)
    
    turbo_state, heater_range = init_sweep_state(
                                            fridge, 
                                            t_sample_ch, 
                                            t_magnet_ch, 
                                            magnet_active, 
                                            #ramp_rate,
                                            pid_values
    )
    
    meas = Measurement()
    meas.write_period = write_period
    meas.register_parameter(sample_temp)
    meas.register_parameter(parameter_fast)
    params = []
    for param in param_meas:
        if isinstance(param, ParameterBase):
            params.append(param)
            meas.register_parameter(param, setpoints=(sample_temp, parameter_fast))

    with meas.run() as datasaver:

        temperature_setpoints = np.linspace(start_temp, end_temp, points_temp)
        fast_axis_setpoints = np.linspace(start_fast, end_fast, points_fast)

        for temperature_setpoint in temperature_setpoints:
            if magnet_active == True:
                magnet_safe(fridge, t_magnet_ch)
            turbo_state, heater_range = _set_temp_setpoint(
                                                        fridge,
                                                        temperature_setpoint,
                                                        magnet_active,
                                                        t_magnet_ch,
                                                        turbo_state,
                                                        heater_range,
            )
            
            while not _close(sample_temp, temperature_setpoint):
                sleep(0.5)

            current_temperature = sample_temp()
            for setpoint_fast in fast_axis_setpoints:
                parameter_fast(setpoint_fast)
                sleep(1)

                params_get = [(param, param.get()) for param in params]
                datasaver.add_result(
                    (sample_temp, current_temperature), 
                    (parameter_fast, parameter_fast())
                    *params_get
                )
                sleep(wait_time_fast)

            sleep(wait_time_temp)

        return datasaver.dataset