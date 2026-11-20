"""rce-sentinel: cost-aware RCE scanner, git hooks, and zero-day intel agent."""

import os

__version__ = "0.2.0"

# Trust the OS certificate store, so TLS-inspecting proxies (Zscaler and similar)
# that the OS already trusts also work here. RCE_SENTINEL_SYSTEM_CERTS=0 opts out.
if os.environ.get("RCE_SENTINEL_SYSTEM_CERTS", "1").lower() not in {"0", "false", "off"}:
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
