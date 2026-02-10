build:
	@echo "Building Docker image..."
	@build_args=""
	@if [ -f .env ]; then \
		env_lines="$$(cat .env | sed 's/#.*//g' | grep -v '^\$$' )"; \
		export $$(echo "$$env_lines" | xargs); \
		echo "Loaded and exported variables from .env"; \
		vars="$$(echo "$$env_lines" | cut -d'=' -f1)"; \
		for var in $$vars; do \
			build_args="$$build_args --build-arg $$var"; \
		done; \
		echo "Passing build args for: $$vars"; \
	fi; \
	docker build $$build_args -t mlops-example/python:3.12 .
