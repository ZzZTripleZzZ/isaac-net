"""Real-network measurement tools: parsers for srsRAN Project and OpenAirInterface gNB logs, a timestamped UDP
probe, a one-way-delay extractor, the unified schema they all write, and the calibration hooks that turn a
campaign into an NRConfig preset. The protocol they serve is docs/measurement-protocol.md.

Modules: schema (tables), pcap (reader), macnr (MAC-NR framing), srsran, oai (parsers), probe (sender/receiver),
owd (delay extraction), ingest (run manifests -> tables), replay (engine replay), calibrate (fits + preset),
preset (load/write preset files). Command lines: ``python -m isaaclab_net.tools.measure.{probe,ingest,calibrate}``.
"""
