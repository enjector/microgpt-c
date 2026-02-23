declare function is_temperature_hot(temperature: number): boolean;
declare function is_pressure_high(pressure: number): boolean;

function eval(temperature: number, pressure: number): boolean {
    return is_temperature_hot(temperature) && is_pressure_high(pressure);
}
