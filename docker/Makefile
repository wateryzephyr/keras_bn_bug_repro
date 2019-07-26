.PHONY : build run clean

DOCKER_REPO ?= keras_bn
DOCKER_TAG ?= debug
UID:=`id -u`
GID:=`id -g`

build:
	docker build --pull \
		-f Dockerfile \
		-t $(DOCKER_REPO):$(DOCKER_TAG) \
		--build-arg USER_NAME=$(shell id -un) \
		--build-arg USER_ID=$(shell id -u) \
		--build-arg GROUP_NAME=$(shell id -gn) \
		--build-arg GROUP_ID=$(shell id -g) \
		.

run:
	docker run -it --runtime=nvidia --net=host --pid=host --rm \
		-v $(shell pwd)/..:/keras_debug \
		-v /tmp/X11-unix:/tmp/X11-unix \
		-u $(UID):$(GID) \
		$(DOCKER_REPO):$(DOCKER_TAG)

clean:
	docker rmi $(DOCKER_REPO):$(DOCKER_TAG)
