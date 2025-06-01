import rclpy
from action_cycle_controller.action_manager import ActionManager
from exodapt_robot_interfaces.action import ActionDecision
from exodapt_robot_interfaces.srv import State
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from .action_registry import ActionRegistry


class ActionCycleController(Node):
    '''
    '''

    def __init__(self):
        super().__init__('action_cycle_controller')

        self.declare_parameter('ac_freq', 1.0)
        self.declare_parameter('actions_config', 'config/actions.yaml')

        self.ac_loop_freq = float(self.get_parameter('ac_freq').value)
        action_config_pth = self.get_parameter('actions_config').value
        # self.declare_parameter('num_short_chunks', 20)
        # self.num_short_chunks = self.get_parameter('num_short_chunks').value

        self.state = ''
        self.state_sub = self.create_subscription(
            State,
            'state',
            self.update_state_callback,
            10,
        )

        # Timer for action cycle [s]
        self.timer = self.create_timer(self.ac_loop_freq,
                                       self.run_action_cycle)

        # Used to prevent spamming state with high-freq. 'do nothing' actions
        self.prev_decision = None  # TODO Needed ???

        ####################
        #  Action manager
        ####################

        self.get_logger().info('Initializing ActionManager')
        self.action_event_pub = self.create_publisher(
            String,
            'action_event',
            10,
        )
        self.action_running_pub = self.create_publisher(
            String,
            'action_running',
            10,
        )

        self.action_manager = ActionManager(
            self,
            self.action_event_pub,
            self.action_running_pub,
        )

        #######################
        # #  Action registry
        # #####################

        self.action_registry = ActionRegistry(self, self.action_manager)
        self.action_registry.load_from_config(action_config_pth)

        self.get_logger().info(
            f"Loaded {len(self.action_registry.get_valid_actions())} actions")

        #####################
        #  Action decision
        #####################

        self._ad_action_client = ActionClient(
            self,
            ActionDecision,
            'action_decision_action_server',
        )

    def run_action_cycle(self):
        '''
        Runs an action cycle represented by a sequence of callback functions.

        1. Get current state
        --> 2. Get action decision
            --> 3. Execute action decision
        '''
        ad_goal = ActionDecision.Goal()
        ad_goal.state = self.state

        self._ad_action_client.wait_for_server()
        self.ad_goal_future = self._ad_action_client.send_goal_async(ad_goal)
        self.ad_goal_future.add_done_callback(self.ad_response_callback)

    def ad_response_callback(self, ad_goal_future):
        """"""
        ad_goal_handle = ad_goal_future.result()

        if not ad_goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.ad_result_future = ad_goal_handle.get_result_async()
        self.ad_result_future.add_done_callback(self.ad_result_callback)

    def ad_result_callback(self, ad_result_future):
        '''
        '''
        ad_result = ad_result_future.result().result
        action_decision = ad_result.pred_action

        self.get_logger().info(f'Received action: {action_decision}')

        self.execute_action(action_decision)

    def execute_action(self, action_decision: str):
        """Execute actions through the action registry."""
        # Get first character out of a potential sequence
        if len(action_decision) > 0:
            action_decision = action_decision[0]

        # Action validity check
        if action_decision not in self.valid_actions_set:
            self.get_logger().info(
                f'Invalid action received: {action_decision}')
            action_decision = 'a'
            self.get_logger().info(f'==> Do nothing: {action_decision}')

        # Execute action
        success = self.action_registry.execute_action(
            action_decision,
            self.state,
        )
        if not success:
            # TODO Replace with human-readable action name
            self.get_logger().error(
                f'Failed to execute action: {action_decision}')

        self.prev_decision = action_decision

    def update_state_callback(self, msg):
        """ Callback function to update the state of the robot.
        """
        self.state = msg.data


def main(args=None):
    rclpy.init(args=args)

    state_client = ActionCycleController()
    state_client.get_logger().info('Action Cycle Controller is running...')
    rclpy.spin(state_client)

    state_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
