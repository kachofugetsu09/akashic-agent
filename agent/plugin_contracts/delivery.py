"""兼容入口；投递合同由 `agent.plugin_contracts.delivery_api` 拥有。

本模块保留 `agent.plugin_contracts.delivery` 这个较短的别名路径，方便插件引用。
"""

from agent.plugin_contracts.delivery_api import (
    DELIVERY,
    DELIVERY_SENDERS,
    FINAL_OUTPUT_DELIVERY,
    DeliveryAdmissionPort,
    DeliverySendersPort,
    DeliveriesPort,
    FinalOutputDeliveryPort,
    FinalOutputWaiter,
    OpenSender,
    Receipt,
    Sender,
    Sink,
    Status,
    Text,
)

__all__ = [
    "DELIVERY",
    "DELIVERY_SENDERS",
    "FINAL_OUTPUT_DELIVERY",
    "DeliveryAdmissionPort",
    "DeliverySendersPort",
    "DeliveriesPort",
    "FinalOutputDeliveryPort",
    "FinalOutputWaiter",
    "OpenSender",
    "Receipt",
    "Sender",
    "Sink",
    "Status",
    "Text",
]
