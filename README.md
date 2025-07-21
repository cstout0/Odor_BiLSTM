# Odor_BiLSTM

load_intan_rhd_format_REQUIRED is required to transform the .RHD files into .JSON.gz (zipped JSON files). Place into a safe directory where python can access but cannot change.
Load_RHD_Format files are intended for using the load_intan_rhd_format_REQUIRED module to change the RHD files to those JSON.gz files.

Do not use the Read_INtan_RHD_RMS - OLD file, it does not work and is used simply as a backup.

Run_Model is used for the random forest when a model is created and saved, it is used to apply itself to other data rather than having to retrain a model.

BiLSTM file is used when having spike sorted data formated into the SS_BASELINESUB format. I have included a MATTable as an example.

RF_RMS is a random forest applying the RMS to itself, RandomForest is using voltage trace. Make sure you are using the proper Load_RHD_Format (either RMS or not) for the random forest model to match formats, otherwise there will be major accuracy errors.
