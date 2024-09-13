import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from robot_action_interfaces.action import ActionDecision
from robot_reply_interfaces.action import ReplyAction
from robot_state_interfaces.srv import State
from std_msgs.msg import String


class ActionController(Node):
    '''
    '''

    def __init__(self):
        super().__init__('action_manager')

        self.declare_parameter('ac_loop_freq', 1.0)
        self.ac_loop_freq = float(self.get_parameter('ac_loop_freq').value)

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

        self._reply_action_client = ActionClient(self, ReplyAction,
                                                 'reply_action_server')
        self._reply_action_send_goal_future = None

        # Action response publisher
        self.action_resp_pub = self.create_publisher(
            String,
            'action_response',
            10,
        )

        # Timer for action cycle [s]
        self.timer = self.create_timer(self.ac_loop_freq,
                                       self.run_action_cycle)

        # Used to prevent spamming state with high-freq. 'do nothing' actions
        self.prev_action = None

        self.valid_actions = {
            'a': 'do_nothing',
            'b': 'reply',
        }

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

        # State of current action cycle
        self.current_state = state

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

        self.execute_action(pred_action)

    def execute_action(self, pred_action: str):
        '''
        '''

        # TODO How to handle action execution
        # 1. Select appropriate action wrapper
        # 2. Action wrapper executes action
        # 3. Action wrapper returns action response
        # 4. Action response is published to /action_response
        # 5. Callback chain terminates

        valid_action_set = list(self.valid_actions.keys())

        if pred_action not in valid_action_set:
            self.get_logger().info(f'Invalid action received: {pred_action}')
            pred_action = 'a'  # np.random.choice(valid_action_set, p=[0.5, 0.5])
            self.get_logger().info(f'==> Do nothing: {pred_action}')

        if pred_action == 'a':
            # TODO Do nothing action
            self.get_logger().info('Executing action a')
            # Don't publish sequantial 'do nothing' actions
            if pred_action == 'a' and self.prev_action == 'a':
                return
            msg = String()
            msg.data = "Robot action: Do nothing"
            self.action_resp_pub.publish(msg)

        elif pred_action == 'b':
            # TODO Handle reply if already speaking using /is_speaking topic
            self.get_logger().info('Executing action b')
            goal = ReplyAction.Goal()
            goal.state = self.current_state
            self.reply_action_send_goal_future = self._reply_action_client.send_goal_async(
                goal)
            # self.reply_action_done_callback.add_done_callback(
            #     self.action_done_callback)

        else:
            self.get_logger().error(
                f'Undefined behavior for action: {pred_action}')
            # action_resp = f'Undefined behavior: {pred_action}'

        # msg = String()
        # msg.data = action_resp
        # self.action_resp_pub.publish(msg)

        self.prev_action = pred_action

        # Execute action
        # if action == 'action_1':
        #     self.get_logger().info('Executing action 1')
        # elif action == 'action_2':
        #     self.get_logger().info('Executing action 2')
        # else:
        #     self.get_logger().info('No action to execute')

    # def reply_action_done_callback(self, future):
    #     '''
    #     '''
    #     self.reply_action_send_goal_future = None


def main(args=None):
    rclpy.init(args=args)

    state_client = ActionController()
    state_client.get_logger().info('Action controller is running...')
    rclpy.spin(state_client)

    state_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
