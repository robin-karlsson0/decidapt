from exodapt_robot_interfaces.action import ActionReply

from ..base_action import BaseAction


class ReplyAction(BaseAction):
    """Reply action plugin that uses ActionManager"""

    def __init__(self, node, action_manager, config):
        super().__init__(node, action_manager, config)

    def execute(self, state: str):
        goal = ActionReply.Goal()
        goal.state = state
        # Use action server name from config
        server_name = self.config.get('action_server', 'reply_action_server')
        self.action_manager.submit_action(server_name, goal)

    def get_action_key(self) -> str:
        return 'b'

    def get_action_name(self) -> str:
        return 'Reply action'
