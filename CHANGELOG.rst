Changelog
=========

[0.1.5] - 2026-08-XX
--------------------

Added
~~~~~
* Redo logic for saving to zarr to allow for overwriting and check if zarr_output_file_exists
* Speed up gridding by loading into memory
* Add verbose argument to check_time_overlap_between_gridded_and_gauges
* Add workflow_start and end_dates argument
* Express min dist as km instead (to save a lot of memory)


[0.1.4] - 2026-08-19
--------------------

Added
~~~~~
* Add flush=True to print statements for batch computing
* Check for time overlap for gauge grid correlator
* Expect zarr version 3.0.8 or later


[0.1.3] - 2026-08-19
--------------------

Added
~~~~~
* Add object_store_config for GriddedRainfallConfig so you can load from object store instead of file
* Fix error with loading in too much xarray data
* Add check_time_overlap_between_gridded_and_gauges function to data_formatting
* Change schema so GriddedRainfallConfig checks for overlap
* Also allows gridded data to be multiple file paths


[0.1.2] - 2026-06-30
--------------------

Added
~~~~~
* Fix logic for write zarr so that it works properly


[0.1.1] - 2026-06-26
--------------------

Added
~~~~~
* Fix bug with how all_days is looped through to generate the zarr
* Added try_parse_hive_dates to read_ and scan_parquet
* Added way to read in parquet or csv files as rainfall data path to WorkflowConfig basemodel
* Added skip for gridding in workflow if time step not in rainfall data input 

[0.1.0] - 2026-04-27
--------------------

Added
~~~~~
* Add CEHGEARSubDailyProducer (part 4 of 4)
* Fix bugs with CEH_GEAR_workflows
* Prepare very bare bones docs and readme in Sphinx

[0.0.5] - 2026-04-27
--------------------

Added
~~~~~
* Add GaugeVsGridCorrelator (part 3 of 4)
* Fix bugs with QualityController

[0.0.4] - 2026-04-27
--------------------

Added
~~~~~
* Edit nearby data loader to work with in memory df or file paths
* Add QualityController and QCSummariser classes (part 2 of 4) 

[0.0.3] - 2026-04-26
--------------------

Added
~~~~~
* Add pydantic BaseModels for CEH-GEAR workflows
* Plug DataPreparer to the CEH-GEAR subdaily workflow (part 1 of 4)

[0.0.2] - 2026-04-22
--------------------

Added
~~~~~
* DataPreparer, xarray_utils & batch_saving_utils

[0.0.1] - 2026-04-22
--------------------

Added
~~~~~
* Create project via cookiecutter-pypackage.
* Add initial files and layout workflow
