process:
	python temporal_graph/preprocess/p_flight.py
	python temporal_graph/preprocess/p_dblp_v2.py
	python temporal_graph/preprocess/p_enron.py
	python temporal_graph/preprocess/p_GDELTLite.py

backup:	
	python test_backup.py --input_fold data_strict/dblp --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📄 thesis/📦 data/pyg/dblp_strict"
	python test_backup.py --input_fold data_strict/eed --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📄 thesis/📦 data/pyg/eed_strict"
	python test_backup.py --input_fold data_strict/GDELTLite --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📄 thesis/📦 data/pyg/GDELTLite_strict"
	python test_backup.py --input_fold data_strict/tgbl-flight --gdrive_path "/Users/hsiangjenli/Google Drive/My Drive/🏫 2022 NTUST/📄 thesis/📦 data/pyg/tgbl-flight_strict"

all: process backup