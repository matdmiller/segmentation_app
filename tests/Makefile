IMAGE_NAME ?= segmentation-app-e2e
DOCKER ?= docker
DOCKERFILE ?= Dockerfile.e2e
CONTEXT ?= ..
SECRETS_DIR ?= $(abspath $(CONTEXT)/.secrets)
RUN_OPTS ?=

.PHONY: e2e e2e-clean

e2e:
	$(DOCKER) build -t $(IMAGE_NAME) -f $(DOCKERFILE) $(CONTEXT)
	$(DOCKER) run --rm \
		-e MODAL_TOKEN_ID \
		-e MODAL_TOKEN_SECRET \
		-e HUGGINGFACE_TOKEN \
		-e HF_TOKEN \
		-e E2E_SKIP_MODAL \
		-e E2E_SEGMENT_TIMEOUT_MS \
		-v $(SECRETS_DIR):/app/.secrets:ro \
		$(RUN_OPTS) \
		$(IMAGE_NAME)

e2e-clean:
	$(DOCKER) build --no-cache -t $(IMAGE_NAME) -f $(DOCKERFILE) $(CONTEXT)
	$(DOCKER) run --rm \
		-e MODAL_TOKEN_ID \
		-e MODAL_TOKEN_SECRET \
		-e HUGGINGFACE_TOKEN \
		-e HF_TOKEN \
		-e E2E_SKIP_MODAL \
		-e E2E_SEGMENT_TIMEOUT_MS \
		-v $(SECRETS_DIR):/app/.secrets:ro \
		$(RUN_OPTS) \
		$(IMAGE_NAME)
