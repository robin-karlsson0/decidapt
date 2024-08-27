import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from robot_action_interfaces.action import ActionDecision
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

        self.state_req = State.Request()

        ####################
        #  Action servers
        ####################
        self._ac_action_client = ActionClient(self, ActionDecision,
                                              'action_decision_action_server')

        self.ac_goal = ActionDecision.Goal()

        # Action response publisher
        self.action_resp_pub = self.create_publisher(
            String,
            'action_response',
            10,
        )

        # Timer for action cycle
        self.timer = self.create_timer(5, self.run_action_cycle)

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
        state_future.add_done_callback(self.send_ac_goal)

    def send_ac_goal(self, state_future):
        '''
        '''
        # Unpack state response
        state = state_future.result().out_state

        # Create action decision goal
        ac_goal = ActionDecision.Goal()
        ac_goal.state = state

        self._ac_action_client.wait_for_server()
        self.send_ac_goal_future = self._ac_action_client.send_goal_async(
            ac_goal)
        self.send_ac_goal_future.add_done_callback(
            self.ac_goal_response_callback)

    def ac_goal_response_callback(self, ac_future):
        '''
        '''
        goal_handle = ac_future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_ac_result_future = goal_handle.get_result_async()
        self.get_ac_result_future.add_done_callback(
            self.get_ac_result_callback)

    def get_ac_result_callback(self, ac_future):
        '''
        '''
        ac_result = ac_future.result().result
        pred_action = ac_result.pred_action

        self.get_logger().info(f'Received action: {pred_action}')

    def execute_action(self, goal_future):
        '''
        '''
        try:
            goal_handler = goal_future.result()
            action = action_resp.pred_action
            self.get_logger().info(f'Received action: {action}')

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
