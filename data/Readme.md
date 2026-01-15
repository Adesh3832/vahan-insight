# Indian Vehicle Registration Data (RTO-style)

This dataset contains vehicle registration statistics derived from
Indian RTO-style administrative records (FY 2020 onwards).

## Files Included

### 1. vehicle_registrations_500k.csv
A 500,000 row random sample of vehicle registrations.
Suitable for exploratory data analysis and modeling.

### 2. state_month_fuel_aggregated.csv
Aggregated vehicle counts by:
- State
- Registration Month
- Fuel Type

This file is ideal for time-series analysis and trend studies.

## Columns

- registrationYear: Year of vehicle registration
- financialYear: Financial year
- registrationMonthMMYY: Registration month (MM-YY)
- makerName: Vehicle manufacturer
- stateName: State of registration
- rtoCode: RTO code
- rtoName: RTO name
- vehicleCategoryName: Vehicle category
- vehicleModelName: Model name
- fuelName: Fuel type
- vehicleClassName: Vehicle class
- grossVehicleWeight: Gross vehicle weight in kg
- pollutionNorm: Emission norm (e.g., BS-VI)
- saleType: New / Used
- vehicleCount: Number of vehicles registered

## Important Notes

- grossVehicleWeight = 0 indicates that the weight was
  not applicable or not recorded.
- This is commonly observed for two-wheelers and
  certain non-commercial vehicle classes.
- Users are encouraged to explicitly handle this
  value during analysis.

## Suggested Use Cases

- EV adoption trends in India
- Fuel transition analysis
- State-wise vehicle growth
- Time-series forecasting
- Policy and emission norm impact studies
