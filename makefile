run_black:
	python3 -m black . -l 119

run_server:
	python app

run_client:
	streamlit run app/frontend.py

run_app: run_server run_client

run_assignment_tests:
	poetry run pytest assignments/app_test.py