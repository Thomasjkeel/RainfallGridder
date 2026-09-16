from pathlib import Path


def get_ceh_gear_15m_CEH_GEAR_based_kwargs():
    return {
        "gridded_rainfall_col": "rainfall_amount",
        "output_dir": Path("ceh_gear_15m_CEH_GEAR_based"),
        "output_rainfall_name": "rainfall",
        "rainfall_offset_hours": 10,
        "n_hours": 96,
        "min_n_timesteps": 96,  # TODO: change
        "time_res": "15m",
        "min_n_neighbours": 7,
        "n_closest_neighbours": 10,
        "qc_framework": "intenseqc_w_subhourlyqc_rulebase",
        "nearby_rainfall_data_loader_kwargs": {
            "distance_threshold_km": 50,
            "n_closest_neighbours": 10,  # TODO: how many neighbours are needed for QC check
        },
    }


def get_ceh_gear_15m_HadUK_Grid_based_kwargs():
    return {
        "gridded_rainfall_col": "rainfall",
        "output_dir": Path("ceh_gear_15m_HadUK_Grid_based"),
        "output_rainfall_name": "rainfall",
        "rainfall_offset_hours": 9,
        "n_hours": 96,
        "min_n_timesteps": 96,  # TODO: change
        "time_res": "15m",
        "n_closest_neighbours": 10,
        "qc_framework": "intenseqc_w_subhourlyqc_rulebase",
        "nearby_rainfall_data_loader_kwargs": {
            "distance_threshold_km": 50,
            "n_closest_neighbours": 10,  # TODO: how many neighbours are needed for QC check
        },
    }
