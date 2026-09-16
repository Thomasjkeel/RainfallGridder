from rainfallqc import (
    comparison_checks,
    gauge_checks,
    neighbourhood_checks,
    timeseries_checks,
)

intenseqc_framework = {
    "QC2": {
        "function": gauge_checks.check_years_where_annual_mean_k_top_rows_are_zero,
    },
    "QC10": {
        "function": comparison_checks.check_exceedance_of_rainfall_world_record,
    },
    "QC11": {
        "function": comparison_checks.check_hourly_exceedance_etccdi_rx1day,
    },
    "QC12": {
        "function": timeseries_checks.check_dry_period_cdd,
    },
    "QC13": {
        "function": timeseries_checks.check_daily_accumulations,
    },
    "QC14": {
        "function": timeseries_checks.check_monthly_accumulations,
    },
    "QC15": {
        "function": timeseries_checks.check_streaks,
    },
    "QC17": {
        "function": neighbourhood_checks.check_wet_neighbours,
    },
    "QC19": {
        "function": neighbourhood_checks.check_dry_neighbours,
    },
    "QC20": {
        "function": neighbourhood_checks.check_monthly_neighbours,
    },
}

INTENSEQC_W_SUBHOURLYQC_RULEBASE = {
    "QC2": {
        "function": gauge_checks.check_years_where_annual_kth_largest_value_is_zero,
    },
    "QC10": {
        "function": comparison_checks.check_exceedance_of_rainfall_world_record,
    },
    "QC11": {
        "function": comparison_checks.check_hourly_exceedance_etccdi_rx1day,
    },
    "QC12": {
        "function": timeseries_checks.check_dry_period_cdd,
    },
    "QC13": {
        "function": timeseries_checks.check_daily_accumulations,
    },
    "QC14": {
        "function": timeseries_checks.check_monthly_accumulations,
    },
    "QC15": {
        "function": timeseries_checks.check_streaks,
    },
    "QC17": {
        "function": neighbourhood_checks.check_wet_neighbours_hourly,
    },
    "QC19": {
        "function": neighbourhood_checks.check_dry_neighbours_hourly,
    },
    "QC20": {
        "function": neighbourhood_checks.check_monthly_neighbours,
    },
    "HQC_UK1hr": {"function": subhourlyqc_checks.check_exceedance_of_UK_1hr_record},
    "HQC_UK24hr": {"function": subhourlyqc_checks.check_exceedance_of_UK_24hr_record},
    "HQC_UK24hr_rolling": {"function": subhourlyqc_checks.check_daily_exceedance_of_UK_24hr_record},
    "HQC_streaks_20mm": {"function": subhourlyqc_checks.check_streaks_20mm},
    "SHQC_freqResChecker": {"function": subhourlyqc_checks.check_freq_is_subhourly},
    "SHQC_subH_checkr": {"function": subhourlyqc_checks.check_subhourly_thresholds},
}
