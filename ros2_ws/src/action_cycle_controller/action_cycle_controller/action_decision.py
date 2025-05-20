import asyncio
import textwrap
import time

import rclpy
from huggingface_hub import InferenceClient
from exodapt_robot_interfaces.action import LLM
from rclpy.action import ActionClient, ActionServer
from rclpy.node import Node
from exodapt_robot_interfaces.action import ActionDecision
# from robot_pt import action_decision_pt, system_msg_pt


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
            execute_callback=self.execute_callback_tgi,
        )

        self.declare_parameter(
            'llm_action_server', 'llm_action_server_ad_8b_action')
        llm_action_server_name = self.get_parameter('llm_action_server').value
        self._action_client = ActionClient(self, LLM, llm_action_server_name)

        self.declare_parameter('tgi_server_url', 'http://localhost:5000')
        self.tgi_server_url = self.get_parameter('tgi_server_url').value
        base_url = f"{self.tgi_server_url}/v1/"
        self.get_logger().info(f'TGI server url: {base_url}')
        self.client = InferenceClient(
            base_url=base_url,
        )

        # LLM inference params
        self.declare_parameter('max_tokens', 1)
        self.declare_parameter('llm_temp', 0.0)
        self.declare_parameter('llm_seed', 14)
        self.max_tokens = self.get_parameter('max_tokens').value
        self.llm_temp = self.get_parameter('llm_temp').value
        self.llm_seed = self.get_parameter('llm_seed').value

        # All valid actions represented by a single token output
        self.do_nothing_action = 'a'

    async def execute_callback_tgi(self, goal_handle):
        '''
        Callback function that sends the received goal directly to the TGI
        inference server and returns the response as an ActionDecision.Result()
        message.
        '''
        # Unpack ActionDecision.Goal() msg
        goal = goal_handle.request
        state = goal.state

        system_msg = system_msg_pt()
        user_msg = action_decision_pt(state)

        t0 = time.time()

        output = self.client.chat.completions.create(
            model="tgi",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            stream=False,
            max_tokens=self.max_tokens,
            temperature=self.llm_temp,
            seed=self.llm_seed
        )
        pred_action = output.choices[0].message.content

        result = ActionDecision.Result()
        result.pred_action = pred_action

        t1 = time.time()
        dt = t1 - t0

        goal_handle.succeed()
        self.get_logger().info(f"Return '{result.pred_action}' ({dt:.2f} s)")

        # Log prompt and result to file
        ts = self.str_time()
        with open(f'/tmp/{ts}_prompt_ad.txt', 'a') as f:
            f.write(self.log_str(system_msg, user_msg, result.pred_action))

        return result

    async def execute_callback(self, goal_handle):
        '''
        Callback function that sends the received goal to the LLM action server
        and returns the response as an LLM.Result() message.
        '''
        # Unpack ActionDecision.Goal() msg
        goal = goal_handle.request
        state = goal.state

        system_msg = system_msg_pt()
        user_msg = action_decision_pt(state)

        llm_goal = LLM.Goal()
        llm_goal.system_prompt = system_msg
        llm_goal.prompt = user_msg
        llm_goal.max_tokens = self.max_tokens
        llm_goal.temp = self.llm_temp
        llm_goal.seed = self.llm_seed

        t0 = time.time()

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

        t1 = time.time()
        dt = t1 - t0

        goal_handle.succeed()
        self.get_logger().info(f"Return '{result.pred_action}' ({dt:.2f} s)")

        # Log prompt and result to file
        # ts = self.str_time()
        # with open(f'/tmp/{ts}_prompt_ad.txt', 'a') as f:
        #     f.write(self.log_str(system_msg, user_msg, result.pred_action))

        return result

    @staticmethod
    def str_time() -> str:
        '''
        Returns a timestamp in the format '1729514281_1626208'.
        '''
        return str(time.time()).replace('.', '_')

    @staticmethod
    def log_str(sys_msg: str, usr_msg: str, res: str) -> str:
        '''
        Returns a log string representing an LLM prompt and response.
        '''
        log_str = textwrap.dedent('''
            <sys_msg>
            {}
            </sys_msg>
            <usr_msg>
            {}
            </usr_msg>
            <res>
            {}
            </res>
            ''')
        log_str = log_str.format(sys_msg, usr_msg, res)

        # Remove leading/trailing line breaks
        log_str = log_str.strip()

        return log_str


def main(args=None):
    rclpy.init(args=args)

    node = ActionDecisionActionServer()

    asyncio.run(rclpy.spin(node))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
