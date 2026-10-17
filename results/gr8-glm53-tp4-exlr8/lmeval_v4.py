#!/usr/bin/env python3
"""Run lm_eval with IPv4-first DNS (host IPv6 egress to HF CDN is a blackhole)."""
import socket

_orig = socket.getaddrinfo

def _ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
    res = _orig(host, port, family, type, proto, flags)
    res.sort(key=lambda r: 0 if r[0] == socket.AF_INET else 1)
    return res

socket.getaddrinfo = _ipv4_first

import sys
from lm_eval.__main__ import cli_evaluate
sys.exit(cli_evaluate())
