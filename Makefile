SHELL := /bin/bash
.DEFAULT_GOAL := command
.PHONY: command download train

MAKEFILE_PATH := $(abspath $(lastword ${MAKEFILE_LIST}))
PROJECT_PATH := $(dir ${MAKEFILE_PATH})
PROJECT_NAME := $(notdir $(patsubst %/,%,$(dir ${PROJECT_PATH})))

export DOCKER_BUILDKIT ?= 1
export APP_NAME ?= ${PROJECT_NAME}
export KMP_WARNINGS=off

download:
	pipenv run python3 src/entrypoint.py --settings-path src/config/settings.yaml download

train:
	pipenv run python3 src/entrypoint.py --settings-path src/config/settings.yaml train

test:
	pipenv run python3 ${MODULE}

command: build

	docker run \
    --name ${APP_NAME}-command \
    --rm \
    -it \
    -e APP_NAME \
    -v ${PROJECT_PATH}lib:/workspace/lib \
    ${APP_NAME}:latest \
    bash

build:
	docker build -t ${APP_NAME} .