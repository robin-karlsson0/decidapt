import asyncio
import time

import rclpy
from exodapt_robot_interfaces.action import ActionDecision
from exodapt_robot_pt import action_decision_pt
from huggingface_hub import InferenceClient
from rclpy.action import ActionServer
from rclpy.node import Node


class ActionDecisionActionServer(Node):
    '''
    '''

    def __init__(self, **kwargs):
        '''
        '''
        super().__init__('action_decision', **kwargs)

        self.action_server_name = 'action_decision_action_server'

        self._action_server = ActionServer(
            self,
            ActionDecision,
            self.action_server_name,
            execute_callback=self.execute_callback_tgi,
        )

        # LLM inference params
        self.declare_parameter('tgi_server_url', 'http://localhost:5000')
        self.declare_parameter('max_tokens', 1)  # NOTE: Single token output
        self.declare_parameter('llm_temp', 0.0)
        self.declare_parameter('llm_seed', 14)
        self.tgi_server_url = self.get_parameter('tgi_server_url').value
        self.max_tokens = self.get_parameter('max_tokens').value
        self.llm_temp = self.get_parameter('llm_temp').value
        self.llm_seed = self.get_parameter('llm_seed').value

        base_url = f"{self.tgi_server_url}/v1/"
        self.client = InferenceClient(base_url=base_url)

        # All valid actions represented by a single token output
        self.do_nothing_action = 'a'

        self.get_logger().info('ActionDecisionActionServer initialized\n'
                               'Parameters:\n'
                               f'  TGI server url: {self.tgi_server_url}\n'
                               f'  max_tokens={self.max_tokens}\n'
                               f'  llm_temp={self.llm_temp}\n'
                               f'  llm_seed={self.llm_seed}')

    async def execute_callback_tgi(self, goal_handle):
        '''
        Callback function that sends the received goal directly to the TGI
        inference server and returns the response as an ActionDecision.Result()
        message.
        '''
        # Unpack ActionDecision.Goal() msg
        goal = goal_handle.request
        state = goal.state
        valid_actions = goal.valid_actions

        user_msg = action_decision_pt(state, valid_actions)

        t0 = time.time()

        output = self.client.chat.completions.create(
            model="tgi",
            messages=[
                {
                    "role": "user",
                    "content": user_msg
                },
            ],
            stream=False,
            max_tokens=self.max_tokens,
            temperature=self.llm_temp,
            seed=self.llm_seed)
        pred_action = output.choices[0].message.content

        result = ActionDecision.Result()
        result.pred_action = pred_action

        t1 = time.time()
        dt = t1 - t0

        goal_handle.succeed()
        self.get_logger().info(f"Return '{result.pred_action}' ({dt:.2f} s)")

        return result


def main(args=None):
    rclpy.init(args=args)

    node = ActionDecisionActionServer()

    asyncio.run(rclpy.spin(node))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
