import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from robot_action_interfaces.action import LLM
from robot_state_interfaces.srv import State
from std_msgs.msg import String


class ActionController(Node):
    '''
    '''

    def __init__(self):
        super().__init__('action_manager')

        #################################
        #  Service clients
        #################################
        self.state_client = self.create_client(State, 'get_state')
        while not self.state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.action_decision_client = self.create_client(State, 'pred_action')
        while not self.state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.state_req = State.Request()
        self.action_req = State.Request()

        # Action response publisher
        self.action_resp_pub = self.create_publisher(
            String,
            'action_responese',
            10,
        )

        # Timer for action cycle
        self.timer = self.create_timer(0.5, self.run_action_cycle)

    def run_action_cycle(self):
        '''
        Runs an action cycle represented by a sequence of callback functions.

        1. Get current state
        --> 2. Get action decision
            --> 3. Execute action decision
        '''
        self.get_current_state()

    def get_current_state(self):
        '''
        '''
        state_future = self.state_client.call_async(self.state_req)
        state_future.add_done_callback(self.get_action_decision)

    def get_action_decision(self, state_future):
        '''
        '''
        try:
            state_resp = state_future.result()
            state = state_resp.out_state
            self.get_logger().info('Received state')

        except Exception as e:
            self.get_logger().info('Service call failed %r' % (e, ))
            return None

        self.action_req.in_state = state
        action_future = self.action_decision_client.call_async(self.action_req)
        action_future.add_done_callback(self.execute_action)

    def execute_action(self, action_future):
        '''
        '''
        try:
            action_resp = action_future.result()
            action = action_resp.out_state
            self.get_logger().info('Received action')

        except Exception as e:
            self.get_logger().info('Service call failed %r' % (e, ))
            return None

        # TODO How to handle action execution
        # 1. Select appropriate action wrapper
        # 2. Action wrapper executes action
        # 3. Action wrapper returns action response
        # 4. Action response is published to /action_response
        # 5. Callback chain terminates

        if action == 'a':
            # TODO Do nothing action
            self.get_logger().info('Executing action a')
            action_resp = "No action"

        elif action == 'b':
            self.get_logger().info('Executing action b')
            action_resp = "Dummy response"

        else:
            self.get_logger().info(f'Invalid action received: {action}')
            action_resp = f'Invalid action received: {action}'

        msg = String()
        msg.data = action_resp
        self.action_resp_pub.publish(msg)

        # Execute action
        # if action == 'action_1':
        #     self.get_logger().info('Executing action 1')
        # elif action == 'action_2':
        #     self.get_logger().info('Executing action 2')
        # else:
        #     self.get_logger().info('No action to execute')


def main(args=None):
    rclpy.init(args=args)

    state_client = ActionController()
    state_client.get_logger().info('Action controller is running...')
    rclpy.spin(state_client)

    state_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
