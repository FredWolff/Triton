from triton import OxfordTriton
from typing import Tuple, Optional, NoReturn, Union
import numpy as np
from time import sleep
import logging

from qcodes.dataset import Measurement
from qcodes.parameters import Parameter, ParameterBase
from qcodes.instrument import Instrument

logger = logging.getLogger('Temperature Sweep')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('~/TemperatureSweepLog.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def T1dMeasurement(
            fridge: OxfordTriton,
            start: float,
            end: float,
            points: int,
            wait_time: float,
            *param_meas,
            pid_values: Tuple[float, float, float]=(10., 20., 0.),
            t_mc_ch: int=8,
            t_magnet_ch: int=13,
            magnet_active: bool=True,
            write_period: float=5.0,
):
    if not isinstance(t_mc_ch, int):
        raise TypeError('t_mc_ch must be an integer.')
    if not isinstance(t_magnet_ch, int):
        raise TypeError('t_magnet_ch must be an integer.')
    if not isinstance(points, int):
        raise TypeError('points must be an integer.')
    
    sample_temp = Temperature('MC Temperature', fridge, t_mc_ch)

    turbo_state, heater_range = _init_sweep_state(
                                            fridge, 
                                            t_mc_ch, 
                                            t_magnet_ch, 
                                            magnet_active, 
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
            if magnet_active == True:
                magnet_temperature = magnet_check(fridge, t_magnet_ch)
                logging.debug(f'Magnet check passed with magnet temperature \
                              T = {magnet_temperature} K')
            turbo_state, heater_range = _set_temp_setpoint(
                                                        fridge, 
                                                        sample_temp,
                                                        setpoint, 
                                                        magnet_active, 
                                                        t_magnet_ch,
                                                        turbo_state,
                                                        heater_range,
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
            parameter_fast: ParameterBase,
            start_fast: float,
            end_fast: float,
            points_fast: int,
            wait_time_fast: float,
            *param_meas,
            pid_values: Tuple[float, float, float] = (10., 20., 0.),
            t_mc_ch: int=8,
            t_magnet_ch: int=13,
            magnet_active: bool=True,
            write_period: float=5.0,
):
    assert type(t_mc_ch) == int
    assert type(t_magnet_ch) == int
    assert type(points_temp) == int
    assert type(points_fast) == int

    sample_temp = Temperature('MC Temperature', fridge, t_mc_ch)
    
    turbo_state, heater_range = _init_sweep_state(
                                            fridge, 
                                            t_mc_ch, 
                                            t_magnet_ch, 
                                            magnet_active, 
                                            pid_values,
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
                magnet_temperature = magnet_check(fridge, t_magnet_ch)
                logging.debug(f'Magnet check passed with magnet temperature \
                 T = {magnet_temperature} K')

            turbo_state, heater_range = _set_temp_setpoint(
                                                        fridge, 
                                                        sample_temp,
                                                        temperature_setpoint, 
                                                        magnet_active, 
                                                        t_magnet_ch,
                                                        turbo_state,
                                                        heater_range,
            )
            
            while not _close(sample_temp(), temperature_setpoint):
                sleep(0.5)

            sleep(wait_time_temp)

            current_temperature = sample_temp()
            for setpoint_fast in fast_axis_setpoints:
                parameter_fast(setpoint_fast) # is the execution blocked until setpoint_fast has been reached?
                sleep(wait_time_fast)

                params_get = [(param, param.get()) for param in params]
                datasaver.add_result(
                    (sample_temp, current_temperature), 
                    (parameter_fast, parameter_fast())
                    *params_get
                )

        return datasaver.dataset


class Temperature(Parameter):
    def __init__(self, name: str, fridge: Instrument, temperature_ch: int, *args, **kwargs):
        self.temperature_measurement = eval(f'fridge.T{temperature_ch}')
        super().__init__(name=name.lower().replace(' ', '_'), unit='K', label=name, *args, **kwargs)

    def get_raw(self):
        return self.temperature_measurement()


def live_configurator(
    fridge: OxfordTriton,
    sample_temp: Parameter,
    future_setpoint: float, 
    heater_range: float,
    turbo_state: str
) -> Tuple[float, float]:
    
    state = {1: 'on', -1: 'off'}
    critical_temp = .8
    critical_speed = 100
    target_state = state[np.sign(critical_temp - future_setpoint)]
    switch_turbo = (turbo_state != target_state)

    _heater_range_curr = fridge._heater_range_curr
    _heater_range_temp = fridge._heater_range_temp
    target_heater_range = _get_best_heater_range(
                                        _heater_range_temp, 
                                        _heater_range_curr, 
                                        future_setpoint
    )
    switch_range = (heater_range != target_heater_range)

    while switch_turbo or switch_range:
        
        if switch_turbo:
            live_state = state[np.sign(critical_temp - sample_temp())]
            turbo_state = _toggle_turbo(
                                    fridge, 
                                    live_state, 
                                    turbo_state, 
                                    future_setpoint, 
                                    sample_temp, 
                                    critical_speed
            )
            switch_turbo = (turbo_state != target_state)

        if switch_range:
            live_range = _get_best_heater_range(
                                    _heater_range_temp, 
                                    _heater_range_curr, 
                                    sample_temp()
            )
            heater_range = _set_heater_range(fridge, live_range, heater_range)
            switch_range = (heater_range != target_heater_range)
        
        sleep(1)
    
    return turbo_state, heater_range


def _toggle_turbo(
        fridge: OxfordTriton, 
        best_state: str, 
        turbo_state: str,
        future_setpoint: float,
        sample_temp: Parameter,
        critical_speed: float,
) -> str:

    if turbo_state != best_state:

        if best_state == 'off':
            fridge.turb1_state(best_state)
            fridge.pid_setpoint(sample_temp())
            
            while fridge.turb1_speed() > critical_speed:
                sleep(1)
            logging.debug(f'Continuing to setpoint {future_setpoint} K now. \
                          System is currently at {sample_temp()} K and \
                          turbo speed is {fridge.turb1_speed()} Hz.')
            fridge.pid_setpoint(future_setpoint)
                
        logging.info('Turbo 1 has been switched ' + best_state)
    
    return best_state


def _get_best_heater_range(
        _heater_range_temp: list, 
        _heater_range_curr: list,
        temperature: float,
) -> float:
    
    for temperature_threshold, current in zip(_heater_range_temp, _heater_range_curr):
        if temperature < temperature_threshold:
            break
        return current


def _set_heater_range(
        fridge: OxfordTriton,
        best_range: float,
        heater_range: float,
) -> float:

    if heater_range != best_range:
        fridge.pid_range(best_range)
        logging.debug(f"Heater range changed to {best_range} mA.")
        return best_range
    return heater_range


def _set_temp_setpoint(
        fridge: OxfordTriton,
        sample_temp: Parameter,
        setpoint: float, 
        magnet_active: bool, 
        t_magnet_ch: int,
        turbo_state: str,
        heater_range: float,
    ) -> None:

    if magnet_active == True:
        magnet_temperature = magnet_check(fridge, t_magnet_ch)
        logging.debug(f'Magnet check passed with magnet temperature \
                      T = {magnet_temperature} K')

    fridge.pid_setpoint(setpoint)
    
    turbo_state, heater_range = live_configurator(
                                            fridge, 
                                            sample_temp, 
                                            setpoint, 
                                            heater_range, 
                                            turbo_state
    )

    return turbo_state, heater_range


def magnet_check(
        fridge: OxfordTriton, 
        t_magnet_ch: int,
    ) -> Union[NoReturn, float]:

    magnet_temp = eval(f'fridge.T{t_magnet_ch}')
    if magnet_temp > 4.6:
        fridge.pid_setpoint(0)
        # magnet should start sweeping down?
        raise ValueError(f'Magnet temperature is {magnet_temp} K. \
                         Cool down or set magnet_active=False and deactivate magnet.')
    return magnet_temp
    

def _set_pid_controller(fridge: OxfordTriton, pid_values: Tuple[float, float, float]) -> None:
    P, I, D = pid_values
    fridge.pid_p(P)
    fridge.pid_i(I)
    fridge.pid_d(D)
    logging.debug(f'PID-values set to: P = {P}, I = {I}, D = {D}')


def _set_active_channels(
                fridge: OxfordTriton,
                t_mc_ch: int,
                t_magnet_ch: int,                
) -> None:
    
    keep_on_channels = [t_mc_ch, t_magnet_ch]
    keep_on_channels = ['T%s' % ch for ch in keep_on_channels]
    for ch in fridge.chan_temps.difference(keep_on_channels):
        eval(f'fridge.' + ch + '_enable("OFF")')
        logging.debug(f'Excitation on temperature channel {ch} \
                      is switched off')


def _init_sweep_state(
                fridge: OxfordTriton,
                t_mc_ch: int,
                t_magnet_ch: int,
                magnet_active: bool,
                pid: Tuple[float, float, float],
    ) -> Tuple[str, str]:

    if magnet_active:
        magnet_temperature = magnet_check(fridge, t_magnet_ch)
        logging.info(f'Magnet check passed with magnet temperature \
                 T = {magnet_temperature} K')

    _set_active_channels(fridge, t_mc_ch, t_magnet_ch)
    _set_pid_controller(fridge, pid)

    return fridge.turb1_state(), fridge.pid_range()


def _close(temperature: float, setpoint: float, tolerance: float=1e-4):
    if abs(temperature - setpoint) < tolerance:
        return True
    else:
        return False