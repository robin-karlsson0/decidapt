import textwrap

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from robot_action_interfaces.action import ActionDecision
from robot_reply_interfaces.action import ReplyAction
from robot_state_interfaces.srv import State
from std_msgs.msg import Bool, String


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
        self.declare_parameter('num_short_chunks', 20)
        self.ac_loop_freq = float(self.get_parameter('ac_loop_freq').value)
        self.num_short_chunks = self.get_parameter('num_short_chunks').value

        #################################
        #  Service clients
        #################################
        self.state_client = self.create_client(State, 'get_state')
        while not self.state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                '\'state_client\' service not available, waiting again...')

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

        #############################
        #  Robot state information
        #############################
        self.create_subscription(
            Bool, '/tts_is_speaking', self.tts_is_speaking_callback, 10)
        self.state_is_speaking = False

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
        self.current_visual_info = self.extract_state_part(
            self.current_state, 'visual_information')
        self.current_state_chunks = self.extract_state_part(
            self.current_state, 'state_chunks', exclude_tag=True)
        self.current_robot_state = self.get_robot_state()

        # Add visual information to state chunk:
        # <visual_information>
        # ...
        # </visual_informmation>
        #
        # State chunk 1
        # ...
        self.current_state_chunks = \
            self.current_state_chunks + \
            '\n\n' + \
            self.current_visual_info + \
            '\n' + \
            self.current_robot_state

        short_state_chunks = self.extract_state_part(
            self.current_state, 'state_chunks', exclude_tag=True)
        short_state_chunks = self.extract_state_chunks(
            short_state_chunks, self.num_short_chunks)
        short_state_chunks = \
            short_state_chunks + \
            '\n\n' + \
            self.current_visual_info + \
            '\n' + \
            self.current_robot_state

        # Create action decision goal (w. shortened state)
        ac_goal = ActionDecision.Goal()
        ac_goal.state = short_state_chunks

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

        # Get first character out of a potential sequence
        if len(pred_action) > 0:
            pred_action = pred_action[0]

        valid_action_set = list(self.valid_actions.keys())

        if pred_action not in valid_action_set:
            self.get_logger().info(f'Invalid action received: {pred_action}')
            pred_action = 'a'
            self.get_logger().info(f'==> Do nothing: {pred_action}')

        if pred_action == 'a':
            # TODO Do nothing action
            # self.get_logger().info('Executing action a')
            # Don't publish sequantial 'do nothing' actions
            if pred_action == 'a' and self.prev_action == 'a':
                return
            msg = String()
            msg.data = "<Robot idle action> Robot decides to take no new action."
            self.action_dec_pub.publish(msg)

        elif pred_action == 'b':

            if self.state_is_speaking:
                self.get_logger().info('Robot still speaking. Ignore')

            if not self.action_manager.is_action_running('reply'):

                msg = String()
                msg.data = "<Robot started reply action> Robot decides to reply to user."
                self.action_dec_pub.publish(msg)

                # self.get_logger().info('Executing action b')
                goal = ReplyAction.Goal()
                goal.state = self.current_state_chunks
                self.reply_action_send_goal_future = self.action_manager.start_action(
                    'reply',
                    self._reply_action_client,
                    goal,
                )
                # NOTE: done ==> 'Send goal' is done (not action complete)
                self.reply_action_send_goal_future.add_done_callback(
                    self.reply_action_response_callback)
            else:
                self.get_logger().info('Reply action already running. Ignore')

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

    def tts_is_speaking_callback(self, msg):
        '''
        Read robot 'is_speaking' robot state.
        '''
        self.state_is_speaking = msg.data

    def get_robot_state(self):
        '''
        Returns a string representing robots state including ongoing actions.
        '''
        robot_state = textwrap.dedent('''
            <robot_state>
            Robot is speaking: {}
            </robot_state>
        ''')
        robot_state = robot_state.format(self.state_is_speaking)
        return robot_state

    @staticmethod
    def extract_state_part(
            state: str,
            tag: str,
            exclude_tag: bool = False,
            ) -> str:
        '''
        Returns a substring representing part of state inside a tag like
            <state_chunks>
            ...
            </state_chunks>.

        Args:
            state: Full state.
            tag: Extract part of state within a tag (ex: state_chunks)
        '''
        # Split the text into lines
        lines = state.split('\n')

        # Find the indices of the last pair of <tag> and </tag> tags
        start_index = -1
        end_index = -1

        for i, line in enumerate(lines):
            if line.strip() == f'<{tag}>':
                start_index = i
            elif line.strip() == f'</{tag}>':
                end_index = i

        if exclude_tag:
            start_index += 1
            end_index -= 1

        # If we found both tags
        if start_index != -1 and end_index != -1 and start_index < end_index:
            # Extract the content between the tags, including the tags themselves
            chunk = '\n'.join(lines[start_index:end_index+1])
            return chunk.strip()
        else:
            return f"No valid {tag} found"

    @staticmethod
    def extract_state_chunks(state_chunks: str, num: int) -> str:
        '''
        Returns the N bottom / most recent state chunks.

        Args:
            state_chunks: Sequence of state information chunks formatted as:
                topic: /asr
                ts: 2024-08-31 17:45:23
                data: <Robot heard voice> User says: Hey robot, I'm heading ...
                ---
                topic: /action_response
                ts: 2024-08-31 17:45:25
                data: <Robot completed reply action> Robot says: Certainly! ...
                ---
            num: Number of chunks to extract counting from the bottom.
        '''
        # Split the input string into chunks
        chunks = state_chunks.strip().split('---')

        # Remove any empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        # Get the N bottom chunks
        bottom_chunks = chunks[-num:]

        # Join the chunks back together with the separator
        result = '\n---\n'.join(bottom_chunks) + '\n---'

        return result

def main(args=None):
    rclpy.init(args=args)

    state_client = ActionController()
    state_client.get_logger().info('Action controller is running...')
    rclpy.spin(state_client)

    state_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
