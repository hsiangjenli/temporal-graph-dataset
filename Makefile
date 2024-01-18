process:
	python temporal_graph/preprocess/p_flight.py
	python temporal_graph/preprocess/p_coin.py
	python temporal_graph/preprocess/p_dblp.py
	python temporal_graph/preprocess/p_enron.py

backup:
	python test_backup.py --input_folder data/tgbl-coin --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/tgbl-coin"
	python test_backup.py --input_folder data/tgbl-flight --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/tgbl-flight"
	python test_backup.py --input_folder data/eed --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/eed"
	python test_backup.py --input_folder data/dblp --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📦 data/pyg/dblp"