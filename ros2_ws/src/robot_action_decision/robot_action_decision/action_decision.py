import threading

import rclpy
from llm_action_interfaces.action import LLM
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robot_state_interfaces.srv import State


class ActionDecision(Node):
    '''
    '''

    def __init__(self):
        super().__init__('action_decision')

        self.callback_group = ReentrantCallbackGroup()

        self.action_decision_service = self.create_service(
            State,
            'pred_action',
            self.pred_action_decision_callback,
        )

        self.llm_action_client = ActionClient(self, LLM, 'llm_action_server')

        # For returning action server result in service callback
        self.result_event = threading.Event()
        self.result = None

    def pred_action_decision_callback(self, request, response):
        '''
        '''
        self.get_logger().info('Received action decision request...')
        self.get_logger().info('Request: {}'.format(request))

        state = request.in_state

        prompt = 'Task: Predict the optimal action to take in order to achieve the goal based on the following robot world state:\n\n'
        prompt += state
        prompt += '\n\nOutput the alphabet character of the optimal action choice in the following list of actions:\n\n'
        prompt += 'Do nothing: a\n'
        prompt += 'Formulate a vocal reply: b\n'
        prompt += 'Character of optimal action choice: '

        goal_msg = LLM.Goal()
        goal_msg.prompt = prompt

        self.llm_action_client.wait_for_server()
        self._send_goal_future = self.llm_action_client.send_goal_async(
            goal_msg)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

        #####################################
        #  Waits for action server results
        #####################################
        # Wait for the result to be set
        self.result_event.wait()

        # Set the response
        response.out_state = self.result

        # Reset the event and result for the next call
        self.result_event.clear()
        self.result = None

        return response

    def goal_response_callback(self, future):
        '''
        '''
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.response))

        # Set the result and signal that it's ready
        self.result = result.response
        self.result_event.set()


def main(args=None):

    rclpy.init(args=args)

    node = ActionDecision()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
