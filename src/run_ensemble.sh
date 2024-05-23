#!/bin/bash

env XLA_FLAGS=--xla_force_host_platform_device_count=8 python3 optimize_ensemble.py
