from triton import OxfordTriton
from typing import Tuple, Optional, NoReturn
import numpy as np


def move_temp_setpoint(fridge: OxfordTriton, set_point: float) -> None:
    fridge.pid_setpoint(set_point)


def toggle_turbo(
        fridge: OxfordTriton, 
        future_setpoint: float, 
        turbo_state: str
    ) -> Optional[str]:
    
    state = {1: 'on', -1: 'off'}
    critcal_temp = .8
    best_state = state[np.sign(critcal_temp - future_setpoint)]

    if best_state != turbo_state:
        fridge.turb1(best_state)
        print('Turbo 1 has been switched ' + best_state)

    
def _set_heater_range(
        fridge: OxfordTriton, 
        future_setpoint: float, 
        current_heater_range: float
    ) -> Optional[str]:

    for temp, curr in zip(fridge._heater_range_temp, fridge._heater_range_curr):
        if future_setpoint < temp:
            if current_heater_range != curr:
                fridge.pid_range(curr)
                print(f"Heater range changed to {curr} mA.")
            break


def _set_temp_setpoint(
        fridge: OxfordTriton, 
        set_point: float, 
        magnet_active: bool, 
        t_sample_ch: int,
        t_magnet_ch: int,
        turbo_state: str,
        heater_range: float,
    ) -> None:

    if magnet_active == True:
        magnet_safe(fridge, t_magnet_ch)
    
    current_setpoint = fridge.pid_setpoint()
    current_temp = eval(f'fridge.T{t_sample_ch}()')


def magnet_safe(
        fridge: OxfordTriton, 
        t_sample_ch: int, 
        t_magnet_ch: int
    ) -> Optional[NoReturn]:

    magnet_temp = eval(f'fridge.T{t_magnet_ch}()')
    if magnet_temp > 4.6:
        move_temp_setpoint(fridge, 0)
        raise ValueError('Magnet temperature is above 4.6K. \
                         Cool down or set magnet_active=False to deactivate magnet.')
    

def set_pid_controller(pid_values: Tuple[float, float, float]) -> None:
    P, I, D = pid_values


def set_ramp_rate(fridge: OxfordTriton, rate: float) -> None:
    fridge.pid_rate(rate) 


def fridge_to_sweep_state(
                        fridge: OxfordTriton,
                        t_sample_ch: int,
                        t_magnet_ch: int,
                        magnet_active: bool,
                        pid: Tuple[float, float, float],
    ) -> Tuple[str, str]:

    if magnet_active:
        magnet_safe(fridge, t_magnet_ch)
    print('Magnet is safe')

    keep_on_channels = [t_sample_ch, t_magnet_ch]
    keep_on_channels = ['T%s' % ch for ch in keep_on_channels]
    for ch in fridge.chan_temps.difference(keep_on_channels):
        eval(f'fridge.' + ch + '_enable("OFF")')

    return fridge.turb1(), fridge.pid_range()


def TMeasurement(
    fridge: OxfordTriton,
    start: float,
    end: float,
    pid_values: Tuple[float, float, float],
    step_interval: float,
    *param_meas,
    t_sample_ch: int=8,
    t_magnet_ch: int=13,
    step_mode: str='time',
    magnet_active: bool=True,
    interval_precision: float=5e-4,
    write_period: float=5.0,
):
    assert type(t_sample_ch) == int
    assert type(t_magnet_ch) == int
    
    turbo_state, heater_range = fridge_to_sweep_state(
                                                    fridge, 
                                                    t_sample_ch, 
                                                    t_magnet_ch, 
                                                    magnet_active, 
                                                    pid_values
    )
    
    meas = Measurement()
    meas.write_period = write_period
    meas.register_parameter(fridge.temperature)
    params = []
    for param in param_meas:
        if isinstance(param, ParameterBase):
            params.append(param)
            meas.register_parameter(param, setpoints=(kiutra.temperature,))

    kiutra.temperature_rate(rate)
    kiutra.temperature(start)
    while not kiutra.temperature.temperature_control.stable:
        time.sleep(0.1)

    with meas.run() as datasaver:
        kiutra.temperature.sweep(start, end, rate)
        stable = False
        T_2 = start
        setpoints_measured = []
        while all((not stable, up_down_condition(T_2, start, end))):
            stable = kiutra.temperature.temperature_control.stable
            T_1 = kiutra.temperature()
            params_get = [(param, param.get()) for param in params]
            T_2 = kiutra.temperature()
            datasaver.add_result((kiutra.temperature, (T_1 + T_2) / 2.0), *params_get)
            # time.sleep(delay)
            setpoints_measured = wait_for_next_setpoint(
                kiutra,
                step_mode,
                start,
                end,
                setpoints_measured,
                step_interval,
                interval_precision
            )

        return datasaver.dataset