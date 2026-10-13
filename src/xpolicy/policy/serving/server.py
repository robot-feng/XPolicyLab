"""Service construction using the existing RPC implementation unchanged."""

from xpolicy.core.config import LoadedContracts, ServerSettings
from xpolicy.policy.runtime import build_policy


def create_server(contracts: LoadedContracts, settings: ServerSettings):
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    return PolicyServer(build_policy(contracts), PolicyServerConfig(
        host=settings.host, port=settings.port,
        ws_ping_interval_s=settings.ws_ping_interval_s,
        ws_ping_timeout_s=settings.ws_ping_timeout_s,
    ))
