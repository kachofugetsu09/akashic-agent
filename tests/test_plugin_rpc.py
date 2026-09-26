"""外部 RPC 的方法、参数和生命周期由实际 provider 拥有。"""
import pytest
from agent.plugin_composition.rpc import rpc_method_key

def test_plugin_method_cannot_replace_host_management():
    with pytest.raises(ValueError, match="已经存在"):
        rpc_method_key("plugin/install")
