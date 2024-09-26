import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from robot_action_interfaces.action import ActionDecision
from robot_reply_interfaces.action import ReplyAction
from robot_state_interfaces.srv import State
from std_msgs.msg import String


class ActionManager:
    '''
    Allows running multiple actions of different names in parallel.

    Dictionary 'running_actions' keep track of running actions by name:
        running_actions[action_name] --> action_client
    '''
    def __init__(self):
        self.running_actions = {}

    def start_action(
            self,
            action_name: str,
            action_client: ActionClient,
            goal,
            ) -> ActionClient:
        '''
        Adds ROS 2 Action Client object to set of running actions and returns
        the Action Goal's future callback function.

        Args:
            action_name: Identifying name of action
            action_client: ROS 2 Action Client object
            goal: Action goal object
        '''
        # TODO Remove? Or update state with "current running actions"?
        if action_name in self.running_actions:
            self.cancel_action(action_name)

        self.running_actions[action_name] = action_client

        action_send_goal_future = action_client.send_goal_async(goal)
        return action_send_goal_future

    def cancel_action(self, action_name: str):
        if action_name in self.running_actions:
            self.running_actions[action_name].cancel_goal_async()
            del self.running_actions[action_name]

    def is_action_running(self, action_name: str):
        return action_name in self.running_actions

    def complete_action(self, action_name: str):
        '''
        '''
        if action_name in self.running_actions:
            del self.running_actions[action_name]


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
            self.get_logger().info('\'state_client\' service not available, waiting again...')

        self.state_req = State.Request()


        ####################
        #  Action servers
        ####################
        self._ac_action_client = ActionClient(self, ActionDecision,
                                              'action_decision_action_server')

        self._reply_action_client = ActionClient(self, ReplyAction,
                                                 'reply_action_server')

        # Action decision publisher
        self.action_dec_pub = self.create_publisher(
            String,
            'action_decision',
            10,
        )
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

        self.get_logger().info('Initializing ActionManager')
        self.action_manager = ActionManager()

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
            # self.get_logger().info('Executing action a')
            # Don't publish sequantial 'do nothing' actions
            if pred_action == 'a' and self.prev_action == 'a':
                return
            msg = String()
            msg.data = "<Robot idle action> Robot decides to do nothing."
            self.action_dec_pub.publish(msg)

        elif pred_action == 'b':

            msg = String()
            msg.data = "<Robot started reply action> Robot decides to reply to user."
            self.action_dec_pub.publish(msg)

            if self.prev_action == 'b':
                self.get_logger().info('Repeated "b" action !!!')
            
            elif not self.action_manager.is_action_running('reply'):
                # self.get_logger().info('Executing action b')
                goal = ReplyAction.Goal()
                goal.state = self.current_state
                self.reply_action_send_goal_future = self.action_manager.start_action(
                    'reply',
                    self._reply_action_client,
                    goal,
                )
                # NOTE: done ==> 'Send goal' is done (not action complete)
                self.reply_action_send_goal_future.add_done_callback(
                    self.reply_action_response_callback)

        else:
            self.get_logger().error(
                f'Undefined behavior for action: {pred_action}')

        self.prev_action = pred_action

    def reply_action_response_callback(self, future):
        '''
        '''
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('\'Reply\' action goal rejected')
            self.action_manager.complete_action('reply')
            return
        
        self.get_reply_result_future = goal_handle.get_result_async()
        self.get_reply_result_future.add_done_callback(
            self.reply_action_completed_callback)
        
    def reply_action_completed_callback(self, future):
        '''
        '''
        self.action_manager.complete_action('reply')
        self.get_logger().info('\'Reply\' action completed')


def main(args=None):
    rclpy.init(args=args)

    state_client = ActionController()
    state_client.get_logger().info('Action controller is running...')
    rclpy.spin(state_client)

    state_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
