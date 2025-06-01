from std_msgs.msg import String

from ..base_action import BaseAction


class IdleAction(BaseAction):
    """Simple do-nothing idle action plugin"""

    def __init__(self, node, action_manager, config):
        super().__init__(node, action_manager, config)
        self.prev_execution = False

    def execute(self, state: str):
        # Prevent spam logging
        if self.prev_execution:
            return

        msg = String()
        msg.data = "Idle actions: Robot decides to take no new action."
        self.node.action_event_pub.publish(msg)
        self.prev_execution = True

    def get_action_key(self) -> str:
        return 'a'

    def get_action_name(self) -> str:
        return 'Idle action'
