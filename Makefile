process:
	# python temporal_graph/preprocess/p_flight.py
	# python temporal_graph/preprocess/p_coin.py
	# python temporal_graph/preprocess/p_dblp.py
	# python temporal_graph/preprocess/p_enron.py
	python temporal_graph/preprocess/p_dblp_year.py --start_year 1936 --end_year 2021
	python temporal_graph/preprocess/p_dblp_year.py --start_year 1936 --end_year 2022
	python temporal_graph/preprocess/p_dblp_year.py --start_year 1936 --end_year 2023
	python temporal_graph/preprocess/p_dblp_year.py --start_year 1936 --end_year 2024

backup:
	# python test_backup.py --input_folder data/tgbl-coin --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/tgbl-coin"
	# python test_backup.py --input_folder data/tgbl-flight --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/tgbl-flight"
	# python test_backup.py --input_folder data/eed --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/eed"
	# python test_backup.py --input_folder data/dblp --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/dblp"
	# python test_backup.py --input_fold data/dblp_1936-2021 --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/dblp_1936-2021"
	# python test_backup.py --input_fold data/dblp_1936-2022 --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/dblp_1936-2022"
	# python test_backup.py --input_fold data/dblp_1936-2023 --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/dblp_1936-2023"
	# python test_backup.py --input_fold data/dblp_1936-2024 --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/dblp_1936-2024"
	python test_backup.py --input_fold data/dblp_v2 --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/dblp_v2"
all: process backup