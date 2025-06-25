import asyncio
import json
import os
import time
from datetime import datetime

import rclpy
from exodapt_robot_interfaces.action import ActionDecision
from exodapt_robot_pt import action_decision_pt
from huggingface_hub import InferenceClient
from rclpy.action import ActionServer
from rclpy.node import Node


class ActionDecisionActionServer(Node):
    """ROS 2 action server for LLM-based robot action decision making.

    This action server integrates Large Language Models (LLMs) with robotic
    systems to provide intelligent action selection based on current robot
    state and available action options. It serves as a bridge between ROS 2
    action clients and Text Generation Inference (TGI) servers, enabling
    natural language processing capabilities for autonomous decision making.

    The server operates by receiving action decision requests containing robot
    state information and valid action lists, formatting these into prompts
    for the LLM, and returning single-token action predictions. This design
    ensures low-latency responses suitable for real-time robotic applications.

    Architecture:
        - Receives ActionDecision goals containing state and valid actions
        - Formats inputs using action_decision_pt prompt template
        - Queries TGI server with HuggingFace InferenceClient
        - Returns single-character action decisions
        - Optional logging of prediction input/output pairs

    Key Features:
        - Single-token output for fast inference and deterministic parsing
        - Configurable LLM parameters (temperature, seed, max tokens)
        - Optional prediction logging for model evaluation and debugging
        - Asynchronous execution with ROS 2 action server architecture
        - Integration with TGI servers for scalable LLM deployment

    Parameters:
        action_server_name (str): Name of the action server
                                 (default: 'action_decision_action_server')
        log_pred_io_pth (str): Directory path for logging LLM prediction
                              input/output pairs. If empty, no logging is
                              performed (default: '')
        tgi_server_url (str): Base URL of the TGI server
                             (default: 'http://localhost:8000')
        max_tokens (int): Maximum tokens for LLM generation
                         (default: 1, NOTE: Single token output)
        llm_temp (float): Temperature parameter for LLM sampling
                         (default: 0.0)
        llm_seed (int): Random seed for deterministic LLM output
                       (default: 14)

    Action Server:
        Service: ActionDecision (exodapt_robot_interfaces/action/ActionDecision)
        Goal:
            - state (str): Current robot state description
            - valid_actions (str): Description of available actions
        Result:
            - pred_action (str): Single character representing predicted action

    Example Usage:
        The action server automatically starts upon node initialization and
        waits for ActionDecision goal requests. Clients send robot state and
        valid action descriptions, receiving single-character action predictions
        suitable for robotic control systems.

    Note:
        Requires a running TGI server at the configured URL. The single-token
        output constraint ensures fast inference but requires careful prompt
        engineering to map complex decisions to single characters.
    """

    def __init__(self, **kwargs):
        """Initialize the ActionDecisionActionServer node.

        Sets up the complete LLM-based action decision infrastructure including:
        - ROS 2 node parameters for server configuration and LLM settings
        - ActionServer for handling ActionDecision requests
        - HuggingFace InferenceClient for TGI server communication
        - Optional logging directory creation for prediction I/O tracking

        Args:
            **kwargs: Additional keyword arguments passed to the parent Node
                     constructor for advanced ROS 2 node configuration

        Parameters configured:
            log_pred_io_pth (str): Directory path where LLM prediction
                (input, output) will be logged as individual JSON files. If
                empty, no logging will be performed.
                Ex: 'log/action_decision/'
        """
        super().__init__('action_decision', **kwargs)

        # Action Server params
        self.declare_parameter(
            'action_server_name',
            'action_decision_action_server',
        )
        self.declare_parameter('log_pred_io_pth', '')

        # LLM inference params
        self.declare_parameter('tgi_server_url', 'http://localhost:8000')
        self.declare_parameter('max_tokens', 1)  # NOTE: Single token output
        self.declare_parameter('llm_temp', 0.0)
        self.declare_parameter('llm_seed', 14)

        self.action_server_name = self.get_parameter(
            'action_server_name').value
        self.log_pred_io_pth = self.get_parameter('log_pred_io_pth').value
        self.tgi_server_url = self.get_parameter('tgi_server_url').value
        self.max_tokens = self.get_parameter('max_tokens').value
        self.llm_temp = self.get_parameter('llm_temp').value
        self.llm_seed = self.get_parameter('llm_seed').value

        self._action_server = ActionServer(
            self,
            ActionDecision,
            self.action_server_name,
            execute_callback=self.execute_callback_tgi,
        )

        base_url = f"{self.tgi_server_url}/v1/"
        self.client = InferenceClient(base_url=base_url)

        # All valid actions represented by a single token output
        self.do_nothing_action = 'a'

        # Create log directory
        if self.log_pred_io_pth:
            if not os.path.exists(self.log_pred_io_pth):
                os.makedirs(self.log_pred_io_pth)

        self.get_logger().info(
            'ActionDecisionActionServer initialized\n'
            'Parameters:\n'
            f'  action_server_name: {self.action_server_name}\n'
            f'  log_pred_io_pth: {self.log_pred_io_pth}\n'
            f'  TGI server url: {self.tgi_server_url}\n'
            f'  max_tokens: {self.max_tokens}\n'
            f'  llm_temp: {self.llm_temp}\n'
            f'  llm_seed: {self.llm_seed}')

    async def execute_callback_tgi(self, goal_handle):
        """Execute action decision prediction using TGI server.

        This callback function processes ActionDecision goals by formatting
        the robot state and valid actions into LLM prompts, querying the TGI
        server for predictions, and returning single-token action decisions.

        Args:
            goal_handle: ROS 2 action goal handle containing the ActionDecision
                        request with state and valid_actions fields

        Returns:
            ActionDecision.Result: Result message containing the predicted
                                  action as a single character string

        Process Flow:
            1. Extract state and valid actions from goal request
            2. Format input using action_decision_pt prompt template
            3. Send chat completion request to TGI server
            4. Extract single-token prediction from response
            5. Log execution time and optionally save prediction I/O
            6. Return result with predicted action

        Performance:
            - Times the complete inference process for monitoring
            - Logs prediction results and execution duration
            - Single-token constraint ensures sub-second response times

        Error Handling:
            TGI server communication errors are propagated to the goal handle,
            allowing clients to handle inference failures appropriately.
        """
        # Unpack ActionDecision.Goal() msg
        goal = goal_handle.request
        state = goal.state
        valid_actions = goal.valid_actions

        llm_input = action_decision_pt(state, valid_actions)

        t0 = time.time()

        output = self.client.chat.completions.create(
            model="tgi",
            messages=[
                {
                    "role": "user",
                    "content": llm_input
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

        # Write prediction IO example to file
        if self.log_pred_io_pth:
            await self.log_pred_io(llm_input, pred_action, dt)

        return result

    async def log_pred_io(self, input: str, output: str, dt: float):
        """Log LLM prediction input and output to JSON file.

        Creates timestamped JSON files containing the complete prediction
        context for model evaluation, debugging, and dataset creation. Each
        log entry includes the prompt, prediction, timing information, and
        timestamps for comprehensive tracking.

        Args:
            input (str): The formatted prompt sent to the LLM
            output (str): The predicted action token returned by the LLM
            dt (float): Inference duration in seconds

        File Format:
            JSON files named 'pred_io_{timestamp_ms}.json' containing:
            - ts: Unix timestamp in milliseconds
            - iso_ts: ISO format timestamp for human readability
            - input: Complete LLM prompt string
            - output: Predicted action token
            - dt: Inference duration in seconds

        Error Handling:
            Logging failures are caught and logged as errors without
            interrupting the action decision process, ensuring system
            reliability when logging is non-critical.

        Note:
            Only logs when log_pred_io_pth parameter is configured.
            Files use UTF-8 encoding with pretty-printed JSON formatting.
        """
        try:
            ts = int(time.time() * 1000)  # millisecond precision
            file_name = f'pred_io_{ts}.json'
            file_pth = os.path.join(self.log_pred_io_pth, file_name)

            log_entry = {
                'ts': ts,
                'iso_ts': datetime.now().isoformat(),
                'input': input,
                'output': output,
                'dt': dt,
            }

            with open(file_pth, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.get_logger().error(
                f"Failed to log prediction IO example: {e}")
            return


def main(args=None):
    """Main entry point for the ActionDecisionActionServer node.

    Initializes the ROS 2 system, creates an ActionDecisionActionServer
    instance, and runs the node until shutdown. This function handles the
    complete lifecycle of the action decision action server.

    Args:
        args: Command line arguments passed to rclpy.init() (optional)

    Lifecycle:
        1. Initialize ROS 2 system
        2. Create ActionDecisionActionServer node instance
        3. Log startup message
        4. Enter ROS 2 spin loop to handle callbacks and timers
        5. Clean up resources on shutdown

    The node will continuously execute action cycles at the configured frequency
    until interrupted or shutdown.
    """
    rclpy.init(args=args)

    node = ActionDecisionActionServer()

    asyncio.run(rclpy.spin(node))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
