"""VNDB Ranking+ data pipeline.

Reads the public VNDB database dump, computes partial-order-network (PONet)
and "scientific" rankings, and exports a compact SQL snapshot for Cloudflare D1.
"""

__version__ = "2.0.0"
