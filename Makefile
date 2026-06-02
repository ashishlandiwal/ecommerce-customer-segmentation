.PHONY: install dev test lint run app docker-build docker-run

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check .

run:
	PYTHONPATH=src python -m segmentation.run --output reports

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t ecommerce-customer-segmentation .

docker-run:
	docker run --rm -p 8501:8501 ecommerce-customer-segmentation
