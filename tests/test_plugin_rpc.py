"""外部 RPC 的方法、参数和生命周期由实际 provider 拥有。"""
import pytest
from plugins.gateway.contract import RpcMethod

def test_plugin_method_cannot_replace_host_management():
    with pytest.raises(ValueError, match="已经存在"):
        RpcMethod.key("plugin/install")
