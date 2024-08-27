import asyncio

import rclpy
from llm_action_interfaces.action import LLM
from rclpy.action import ActionClient, ActionServer
from rclpy.node import Node
from robot_action_interfaces.action import ActionDecision


class ActionDecisionActionServer(Node):
    '''
    '''

    def __init__(self):
        '''
        '''
        super().__init__('action_decision')

        self._action_server = ActionServer(
            self,
            ActionDecision,
            'action_decision_action_server',
            execute_callback=self.execute_callback,
        )

        self._action_client = ActionClient(self, LLM, 'llm_action_server')

        # All valid actions represented by a single token output
        self.max_tokens = 1
        self.do_nothing_action = 'a'

    async def execute_callback(self, goal_handle):
        '''
        Callback function that sends the received goal to the LLM action server
        and returns the response as an LLM.Result() message.
        '''
        # Unpack ActionDecision.Goal() msg
        goal = goal_handle.request
        state = goal.state

        # Create action decision prompt
        prompt = 'Task: Predict the optimal action to take in order to achieve the goal based on the following robot world state:\n\n'
        prompt += state
        prompt += '\n\nOutput the alphabet character of the optimal action choice in the following list of actions:\n\n'
        prompt += f'Do nothing: {self.do_nothing_action}\n'
        prompt += 'Formulate a vocal reply: b\n'
        prompt += 'Character of optimal action choice: '

        llm_goal = LLM.Goal()
        llm_goal.prompt = prompt
        llm_goal.max_tokens = self.max_tokens

        # (1) Send goal asynchronously
        send_goal_future = await self._action_client.send_goal_async(llm_goal)

        # Return "do nothing" action if goal is not accepted
        if not send_goal_future.accepted:
            goal_handle.abort()
            result_msg = ActionDecision.Result()
            result_msg.pred_action = self.do_nothing_action
            return result_msg

        # (2) Wait for the result asynchronously
        llm_result = await send_goal_future.get_result_async()

        # Unpack LLM.Result() msg
        llm_result_msg = llm_result.result

        result = ActionDecision.Result()
        result.pred_action = llm_result_msg.response

        goal_handle.succeed()
        self.get_logger().info(f'Return response: {result}')

        return result


def main(args=None):
    rclpy.init(args=args)

    node = ActionDecisionActionServer()

    asyncio.run(rclpy.spin(node))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
